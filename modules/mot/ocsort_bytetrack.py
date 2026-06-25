"""
OC-SORT + ByteTrack 融合跟踪器

核心改进：
  1. 速度预测位置匹配 - 同时使用卡尔曼预测和last_observation，取最小距离
  2. 自适应阈值 - 丢失越久的轨迹，匹配阈值越宽松
  3. 丢失轨迹可匹配低分检测 (Stage 2.5)
  4. 创建新ID前的最终恢复机制 (Stage 4)
  5. 超长track_buffer防止丢失轨迹被过早删除
  6. frame_id同步修复 - 确保last_observation时间戳正确
  7. still_lost包含当前帧新丢失轨迹 - 防止同一帧内创建新ID
  8. 轨迹确认机制 - 新检测必须连续匹配N帧后才分配track_id，过滤误检

匹配阶段：
  阶段一：OC-SORT级匹配 - confirmed tracked + tentative vs high_detections (IoU + OOM)
  阶段二：中心距离匹配 - lost vs 未匹配high_detections (速度预测+自适应阈值)
  阶段二.五：中心距离匹配 - 剩余lost vs low_detections (抢救部分遮挡目标)
  阶段三：ByteTrack级匹配 - 未匹配tracked vs 剩余low_detections (纯IoU)
  阶段四：新轨迹初始化 - 先尝试恢复lost(含当前帧新丢失)，再创建tentative(空间过滤)
  阶段四.五：tentative轨迹确认 - 连续匹配confirm_frames帧后分配track_id
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class OCSORTKalman:
    _shared_R = None
    _shared_Q_pos = None
    _shared_Q_vel = None

    @classmethod
    def set_parameters(cls, R=0.05, Q_pos=0.01, Q_vel=0.0001):
        cls._shared_R = R
        cls._shared_Q_pos = Q_pos
        cls._shared_Q_vel = Q_vel

    def __init__(self):
        self.dim_x = 8
        self.dim_z = 4

        self.F = np.eye(self.dim_x)
        for i in range(self.dim_z):
            self.F[i, i + self.dim_z] = 1

        self.H = np.zeros((self.dim_z, self.dim_x))
        for i in range(self.dim_z):
            self.H[i, i] = 1

        R_val = OCSORTKalman._shared_R if OCSORTKalman._shared_R is not None else 0.05
        Q_pos_val = OCSORTKalman._shared_Q_pos if OCSORTKalman._shared_Q_pos is not None else 0.01
        Q_vel_val = OCSORTKalman._shared_Q_vel if OCSORTKalman._shared_Q_vel is not None else 0.0001

        self.R = np.eye(self.dim_z) * R_val
        self.P = np.eye(self.dim_x) * 10.0
        self.P[4:, 4:] *= 1000.0
        self.Q = np.eye(self.dim_x)
        self.Q[:4, :4] *= Q_pos_val
        self.Q[4:, 4:] *= Q_vel_val

    def initiate(self, measurement):
        mean = np.zeros(self.dim_x)
        mean[:self.dim_z] = measurement
        covariance = self.P.copy()
        return mean, covariance

    def predict(self, mean, covariance):
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + self.Q
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean = self.H @ mean
        projected_cov = self.H @ covariance @ self.H.T + self.R
        kalman_gain = covariance @ self.H.T @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean
        mean = mean + kalman_gain @ innovation
        covariance = covariance - kalman_gain @ self.H @ covariance
        return mean, covariance


class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class STrack:
    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.tracklet_len = 0
        self.state = TrackState.NEW

        self.kf = OCSORTKalman()
        self.mean, self.covariance = self.kf.initiate(self.tlwh_to_xyah(self._tlwh))

        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0

        self.position_history = deque(maxlen=60)
        self.score_history = deque(maxlen=30)

        self.last_observation = None
        self.observation_ages = []
        self.delta_t = deque(maxlen=60)

        self.is_activated = False

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def xyah_to_tlwh(xyah):
        ret = np.asarray(xyah).copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlwh(self):
        return self.xyah_to_tlwh(self.mean[:4])

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def bbox(self):
        return self.tlbr

    def predict(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)

    def update(self, tlwh, score, frame_id=None):
        self._tlwh = tlwh
        self.score = score

        if frame_id is not None:
            self.frame_id = frame_id

        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, self.tlwh_to_xyah(tlwh)
        )
        self.tracklet_len += 1
        self.state = TrackState.TRACKED

        center = np.array([tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2])
        self.position_history.append(center)
        self.score_history.append(score)

        if self.last_observation is not None:
            dt = self.frame_id - self.last_observation[0]
            if dt > 0:
                self.delta_t.append(dt)

        self.last_observation = (self.frame_id, center.copy(), tlwh.copy())
        self.observation_ages.append(self.frame_id)

    def mark_lost(self):
        self.state = TrackState.LOST

    def mark_removed(self):
        self.state = TrackState.REMOVED

    def get_center(self):
        tlwh = self.tlwh
        return np.array([tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2])

    def get_velocity(self):
        if len(self.position_history) < 2:
            return np.zeros(2)
        n = min(5, len(self.position_history))
        positions = list(self.position_history)[-n:]
        if len(positions) < 2:
            return np.zeros(2)
        velocity = np.array(positions[-1]) - np.array(positions[0])
        velocity /= (len(positions) - 1)
        return velocity

    def oc_sort_recover(self, new_tlwh, new_score, lost_frames, frame_id=None):
        if self.last_observation is not None:
            old_center = self.last_observation[1]
            new_center = np.array([new_tlwh[0] + new_tlwh[2] / 2, new_tlwh[1] + new_tlwh[3] / 2])

            if lost_frames > 0:
                avg_velocity = (new_center - old_center) / lost_frames
                self.mean[4] = avg_velocity[0]
                self.mean[5] = avg_velocity[1]
                self.kf.R = np.eye(4) * 0.1

        self.update(new_tlwh, new_score, frame_id=frame_id)
        self.state = TrackState.TRACKED


class OCSORTByteTracker:
    def __init__(self,
                 track_thresh: float = 0.50,
                 match_thresh: float = 0.80,
                 track_buffer: int = 30000,
                 frame_rate: int = 30,
                 min_center_distance: float = 150.0,
                 kalman_R: float = 0.05,
                 kalman_Q_pos: float = 0.01,
                 kalman_Q_vel: float = 0.0001,
                 confirm_frames: int = 3,
                 max_recover_distance: float = 300.0,
                 max_speed_per_frame: float = 10.0):

        OCSORTKalman.set_parameters(kalman_R, kalman_Q_pos, kalman_Q_vel)

        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.min_center_distance = min_center_distance
        self.confirm_frames = confirm_frames
        self.max_recover_distance = max_recover_distance
        self.max_speed_per_frame = max_speed_per_frame

        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

        self._center_dist_thresh = 0.6
        self._center_norm_thresh = 4.0

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.tentative_stracks: List[STrack] = []

        self.frame_id = 0
        self.track_id = 0

    def _get_predicted_center(self, track):
        if track.last_observation is None:
            return track.get_center()

        last_center = track.last_observation[1]
        elapsed = self.frame_id - track.last_observation[0]

        if elapsed <= 0:
            return last_center

        velocity = track.get_velocity()
        if np.linalg.norm(velocity) > 0:
            max_extrap = 15
            extrap_frames = min(elapsed, max_extrap)
            predicted = last_center + velocity * extrap_frames
            return predicted

        return last_center

    def update(self, high_dets: List[Dict], low_dets: List[Dict] = None) -> List[STrack]:
        self.frame_id += 1

        if low_dets is None:
            low_dets = []

        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        high_detections = self._convert_detections(high_dets)
        low_detections = self._convert_detections(low_dets)

        for track in self.tracked_stracks:
            track.predict()
        for track in self.lost_stracks:
            track.predict()
        for track in self.tentative_stracks:
            track.predict()

        # ============================================================
        # 阶段一：confirmed tracked + tentative vs high_detections
        # ============================================================
        tracked_pool = [t for t in self.tracked_stracks if t.state == TrackState.TRACKED]
        tentative_pool = [t for t in self.tentative_stracks if t.state == TrackState.TRACKED]
        all_tracked_pool = tracked_pool + tentative_pool

        if len(all_tracked_pool) > 0 and len(high_detections) > 0:
            dists_tracked = self._compute_oc_sort_distance(all_tracked_pool, high_detections)

            for i, track in enumerate(all_tracked_pool):
                if track.last_observation is None:
                    continue
                last_center = track.last_observation[1]
                for j, det in enumerate(high_detections):
                    det_center = np.array([det['tlwh'][0] + det['tlwh'][2] / 2,
                                           det['tlwh'][1] + det['tlwh'][3] / 2])
                    obs_dist = np.linalg.norm(det_center - last_center)
                    if obs_dist > self.max_recover_distance:
                        dists_tracked[i, j] = 1.0

            matches_tracked, u_track_idx, u_det_idx = self._linear_assignment(
                dists_tracked, thresh=self.match_thresh
            )
        else:
            matches_tracked = []
            u_track_idx = list(range(len(all_tracked_pool)))
            u_det_idx = list(range(len(high_detections)))

        confirmed_activated = []
        tentative_matched = []

        for itracked, idet in matches_tracked:
            track = all_tracked_pool[itracked]
            det = high_detections[idet]
            track.update(det['tlwh'], det['score'], frame_id=self.frame_id)

            if track.is_activated:
                confirmed_activated.append(track)
            else:
                if track.tracklet_len >= self.confirm_frames:
                    self.track_id += 1
                    track.track_id = self.track_id
                    track.is_activated = True
                    confirmed_activated.append(track)
                else:
                    tentative_matched.append(track)

        activated_stracks.extend(confirmed_activated)

        unmatched_tracked = [all_tracked_pool[i] for i in u_track_idx]
        unmatched_confirmed = [t for t in unmatched_tracked if t.is_activated]
        unmatched_tentative = [t for t in unmatched_tracked if not t.is_activated]
        unmatched_high_dets = [high_detections[i] for i in u_det_idx]

        # ============================================================
        # 阶段二：lost轨迹 vs 未匹配high_detections (中心距离+速度预测)
        # ============================================================
        remaining_lost = []
        if len(self.lost_stracks) > 0 and len(unmatched_high_dets) > 0:
            dists_lost = self._compute_center_distance(self.lost_stracks, unmatched_high_dets)
            matches_lost, u_lost_idx, u_det2_idx = self._linear_assignment(
                dists_lost, thresh=self._center_dist_thresh
            )

            for ilost, idet in matches_lost:
                track = self.lost_stracks[ilost]
                det = unmatched_high_dets[idet]
                lost_frames = self.frame_id - track.last_observation[0] if track.last_observation else 0
                if lost_frames > 5:
                    track.oc_sort_recover(det['tlwh'], det['score'], lost_frames, frame_id=self.frame_id)
                else:
                    track.update(det['tlwh'], det['score'], frame_id=self.frame_id)
                refind_stracks.append(track)

            remaining_high_dets = [unmatched_high_dets[i] for i in u_det2_idx]
            remaining_lost = [self.lost_stracks[i] for i in u_lost_idx]
        else:
            remaining_high_dets = unmatched_high_dets
            remaining_lost = list(self.lost_stracks)

        for track in unmatched_confirmed:
            track.mark_lost()
            lost_stracks.append(track)

        for track in unmatched_tentative:
            track.mark_lost()

        # ============================================================
        # 阶段二.五：剩余lost轨迹 vs low_detections (中心距离+速度预测)
        # ============================================================
        remaining_low_dets = low_detections
        if len(remaining_lost) > 0 and len(low_detections) > 0:
            dists_lost_low = self._compute_center_distance(remaining_lost, low_detections)
            matches_lost_low, u_lost2_idx, u_det_low_idx = self._linear_assignment(
                dists_lost_low, thresh=self._center_dist_thresh
            )

            for ilost, idet in matches_lost_low:
                track = remaining_lost[ilost]
                det = low_detections[idet]
                lost_frames = self.frame_id - track.last_observation[0] if track.last_observation else 0
                if lost_frames > 5:
                    track.oc_sort_recover(det['tlwh'], det['score'], lost_frames, frame_id=self.frame_id)
                else:
                    track.update(det['tlwh'], det['score'], frame_id=self.frame_id)
                refind_stracks.append(track)

            remaining_low_dets = [low_detections[i] for i in u_det_low_idx]

        # ============================================================
        # 阶段三：ByteTrack级匹配 - 未匹配tracked vs 剩余low_detections
        # ============================================================
        if len(unmatched_confirmed) > 0 and len(remaining_low_dets) > 0:
            dists_low = self._compute_iou_distance(unmatched_confirmed, remaining_low_dets)

            for i, track in enumerate(unmatched_confirmed):
                if track.last_observation is None:
                    continue
                last_center = track.last_observation[1]
                elapsed = self.frame_id - track.last_observation[0]
                max_allowed = min(
                    self.max_recover_distance,
                    self.max_speed_per_frame * max(elapsed, 1)
                )
                for j, det in enumerate(remaining_low_dets):
                    det_center = np.array([det['tlwh'][0] + det['tlwh'][2] / 2,
                                           det['tlwh'][1] + det['tlwh'][3] / 2])
                    obs_dist = np.linalg.norm(det_center - last_center)
                    predicted = self._get_predicted_center(track)
                    pred_dist = np.linalg.norm(det_center - predicted)
                    if obs_dist > max_allowed and pred_dist > max_allowed:
                        dists_low[i, j] = 1.0

            matches_low, u_track2, u_det_low2 = self._linear_assignment(
                dists_low, thresh=0.5
            )

            for itracked, idet in matches_low:
                track = unmatched_confirmed[itracked]
                det = remaining_low_dets[idet]
                track.update(det['tlwh'], det['score'], frame_id=self.frame_id)
                activated_stracks.append(track)

        # ============================================================
        # 阶段四：新轨迹初始化 - 先尝试恢复lost，再创建tentative
        # ============================================================
        all_remaining_dets = remaining_high_dets
        still_lost = [t for t in self.lost_stracks
                      if t.state == TrackState.LOST and t not in refind_stracks]
        still_lost.extend([t for t in lost_stracks if t.state == TrackState.LOST])

        for det in all_remaining_dets:
            if det['score'] < self.track_thresh:
                continue

            det_center = np.array([det['tlwh'][0] + det['tlwh'][2] / 2,
                                   det['tlwh'][1] + det['tlwh'][3] / 2])

            recovered = self._try_recover_lost(det, det_center, still_lost)
            if recovered is not None:
                refind_stracks.append(recovered)
                still_lost = [t for t in still_lost if t.track_id != recovered.track_id]
                continue

            too_close = self._check_spatial_filter(det_center)
            if too_close:
                continue

            new_track = STrack(det['tlwh'], det['score'])
            new_track.frame_id = self.frame_id
            new_track.start_frame = self.frame_id
            new_track.state = TrackState.TRACKED
            new_track.tracklet_len = 1
            new_track.is_activated = False
            new_track.last_observation = (self.frame_id, det_center.copy(), det['tlwh'].copy())
            new_track.observation_ages.append(self.frame_id)
            self.tentative_stracks.append(new_track)

        # ============================================================
        # 阶段四.五：tentative轨迹确认
        # 连续匹配confirm_frames帧后分配track_id
        # ============================================================
        newly_confirmed = []
        remaining_tentative = []
        for track in self.tentative_stracks:
            if track.state == TrackState.TRACKED and track.is_activated:
                remaining_tentative.append(track)
            elif track.state == TrackState.TRACKED and not track.is_activated:
                if track.tracklet_len >= self.confirm_frames:
                    self.track_id += 1
                    track.track_id = self.track_id
                    track.is_activated = True
                    newly_confirmed.append(track)
                else:
                    remaining_tentative.append(track)
            elif track.state == TrackState.LOST:
                pass
            else:
                pass

        activated_stracks.extend(newly_confirmed)
        self.tentative_stracks = remaining_tentative

        # 超时轨迹删除
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # 更新轨迹列表
        tracked_ids = set()
        unique_tracked = []
        for t in self.tracked_stracks:
            if t.state == TrackState.TRACKED and t.is_activated and t.track_id not in tracked_ids:
                tracked_ids.add(t.track_id)
                unique_tracked.append(t)
        self.tracked_stracks = unique_tracked

        for t in activated_stracks:
            if t.is_activated and t.track_id not in tracked_ids:
                self.tracked_stracks.append(t)
                tracked_ids.add(t.track_id)

        for t in refind_stracks:
            if t.is_activated and t.track_id not in tracked_ids:
                self.tracked_stracks.append(t)
                tracked_ids.add(t.track_id)

        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.LOST]
        self.lost_stracks.extend(lost_stracks)

        self.removed_stracks.extend(removed_stracks)

        return [t for t in self.tracked_stracks if t.state == TrackState.TRACKED and t.is_activated]

    def _try_recover_lost(self, det, det_center, lost_tracks):
        if len(lost_tracks) == 0:
            return None

        best_track = None
        best_cost = float('inf')

        for track in lost_tracks:
            if track.last_observation is None:
                continue

            last_center = track.last_observation[1]
            last_tlwh = track.last_observation[2]
            last_area = last_tlwh[2] * last_tlwh[3]
            elapsed = self.frame_id - track.last_observation[0]

            max_allowed_dist = min(
                self.max_recover_distance,
                self.max_speed_per_frame * max(elapsed, 1)
            )

            predicted_center = self._get_predicted_center(track)

            obs_dist = np.linalg.norm(det_center - last_center)
            pred_dist = np.linalg.norm(det_center - predicted_center)

            if obs_dist > max_allowed_dist and pred_dist > max_allowed_dist:
                continue

            center_dist = min(obs_dist, pred_dist)

            det_area = det['tlwh'][2] * det['tlwh'][3]
            ref_size = max(np.sqrt(last_area), np.sqrt(det_area))
            if ref_size == 0:
                continue

            normalized_dist = center_dist / ref_size

            adaptive_norm = self._center_norm_thresh * (1.0 + 0.01 * min(elapsed, 50))
            adaptive_norm = min(adaptive_norm, self._center_norm_thresh * 2.0)

            if normalized_dist < adaptive_norm:
                cost = normalized_dist / adaptive_norm
                if cost < best_cost:
                    best_cost = cost
                    best_track = track

        if best_track is not None and best_cost < 0.99:
            lost_frames = self.frame_id - best_track.last_observation[0] if best_track.last_observation else 0
            if lost_frames > 5:
                best_track.oc_sort_recover(det['tlwh'], det['score'], lost_frames, frame_id=self.frame_id)
            else:
                best_track.update(det['tlwh'], det['score'], frame_id=self.frame_id)
            return best_track

        return None

    def _check_spatial_filter(self, det_center):
        all_active = self.tracked_stracks + self.lost_stracks + self.tentative_stracks
        for track in all_active:
            if track.state in [TrackState.TRACKED, TrackState.LOST]:
                if track.last_observation is not None:
                    ref_center = track.last_observation[1]
                    predicted = self._get_predicted_center(track)

                    obs_dist = np.linalg.norm(det_center - ref_center)
                    pred_dist = np.linalg.norm(det_center - predicted)

                    min_dist = min(obs_dist, pred_dist)
                else:
                    min_dist = np.linalg.norm(det_center - track.get_center())

                if min_dist < self.min_center_distance:
                    return True
        return False

    def _convert_detections(self, dets: List[Dict]) -> List[Dict]:
        result = []
        for d in dets:
            box = d['box']
            result.append({
                'tlwh': np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]]),
                'score': d['confidence']
            })
        return result

    def _compute_oc_sort_distance(self, tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = self._iou(track.tlbr, det['tlwh'])

                direction_bonus = 0.0
                if len(track.position_history) >= 2 and track.state == TrackState.TRACKED:
                    direction = track.get_velocity()
                    det_center = np.array([det['tlwh'][0] + det['tlwh'][2] / 2,
                                           det['tlwh'][1] + det['tlwh'][3] / 2])
                    track_center = track.get_center()
                    predicted_center = track_center + direction

                    pred_error = np.linalg.norm(det_center - predicted_center)
                    diag = np.sqrt(det['tlwh'][2] ** 2 + det['tlwh'][3] ** 2)
                    if diag > 0:
                        direction_bonus = 0.2 * (1.0 - min(pred_error / diag, 1.0))

                cost_matrix[i, j] = 1.0 - iou - direction_bonus

        return cost_matrix

    def _compute_center_distance(self, tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        cost_matrix = np.ones((len(tracks), len(detections)), dtype=np.float32)

        for i, track in enumerate(tracks):
            if track.last_observation is None:
                continue

            last_center = track.last_observation[1]
            last_tlwh = track.last_observation[2]
            last_area = last_tlwh[2] * last_tlwh[3]
            elapsed = self.frame_id - track.last_observation[0]

            predicted_center = self._get_predicted_center(track)

            max_allowed_dist = min(
                self.max_recover_distance,
                self.max_speed_per_frame * max(elapsed, 1)
            )

            for j, det in enumerate(detections):
                det_center = np.array([det['tlwh'][0] + det['tlwh'][2] / 2,
                                       det['tlwh'][1] + det['tlwh'][3] / 2])
                det_area = det['tlwh'][2] * det['tlwh'][3]

                obs_dist = np.linalg.norm(det_center - last_center)
                pred_dist = np.linalg.norm(det_center - predicted_center)

                if obs_dist > max_allowed_dist and pred_dist > max_allowed_dist:
                    continue

                center_dist = min(obs_dist, pred_dist)

                ref_size = max(np.sqrt(last_area), np.sqrt(det_area))
                if ref_size == 0:
                    continue

                normalized_dist = center_dist / ref_size

                adaptive_norm = self._center_norm_thresh * (1.0 + 0.01 * min(elapsed, 50))
                adaptive_norm = min(adaptive_norm, self._center_norm_thresh * 2.0)

                if normalized_dist < adaptive_norm:
                    cost_matrix[i, j] = normalized_dist / adaptive_norm

        return cost_matrix

    def _compute_iou_distance(self, tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = self._iou(track.tlbr, det['tlwh'])
                cost_matrix[i, j] = 1.0 - iou

        return cost_matrix

    @staticmethod
    def _iou(box1, box2):
        if box2.shape[0] == 4:
            box2 = np.array([box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]])

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    @staticmethod
    def _linear_assignment(cost_matrix, thresh):
        if cost_matrix.size == 0:
            if cost_matrix.ndim == 0:
                return [], [], []
            elif cost_matrix.ndim == 1:
                return [], list(range(cost_matrix.shape[0])), []
            else:
                ua = list(range(cost_matrix.shape[0])) if cost_matrix.shape[0] > 0 else []
                ub = list(range(cost_matrix.shape[1])) if cost_matrix.shape[1] > 0 else []
                return [], ua, ub

        matches = []
        unmatched_a = list(range(cost_matrix.shape[0]))
        unmatched_b = list(range(cost_matrix.shape[1]))

        if cost_matrix.shape[0] > 0 and cost_matrix.shape[1] > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= thresh:
                    matches.append((r, c))
                    if r in unmatched_a:
                        unmatched_a.remove(r)
                    if c in unmatched_b:
                        unmatched_b.remove(c)

        return matches, unmatched_a, unmatched_b

    def get_statistics(self) -> Dict:
        return {
            'active_tracks': len(self.tracked_stracks),
            'lost_tracks': len(self.lost_stracks),
            'removed_tracks': len(self.removed_stracks),
            'tentative_tracks': len(self.tentative_stracks),
            'next_id': self.track_id + 1,
            'max_id_generated': self.track_id
        }
