"""
OC-SORT + ByteTrack 融合跟踪器.

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
import warnings
from collections import deque
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")


class OCSORTKalman:
    """卡尔曼滤波器（针对 OC-SORT 8 维状态空间优化）.

    状态向量: [center_x, center_y, aspect_ratio, height, v_cx, v_cy, v_a, v_h].
    """

    _shared_R = None
    _shared_Q_pos = None
    _shared_Q_vel = None

    @classmethod
    def set_parameters(cls, R: float = 0.05, Q_pos: float = 0.01, Q_vel: float = 0.0001) -> None:
        """设置卡尔曼滤波器的测量噪声与过程噪声超参数.

        Args:
            R: 测量噪声协方差权重，默认 0.05.
            Q_pos: 位置过程噪声权重，默认 0.01.
            Q_vel: 速度过程噪声权重，默认 0.0001.
        """
        cls._shared_R = R
        cls._shared_Q_pos = Q_pos
        cls._shared_Q_vel = Q_vel

    def __init__(self) -> None:
        """初始化状态转移矩阵、观测矩阵与噪声协方差."""
        self.dim_x = 8
        self.dim_z = 4

        self.F = np.eye(self.dim_x)
        for i in range(self.dim_z):
            self.F[i, i + self.dim_z] = 1

        self.H = np.zeros((self.dim_z, self.dim_x))
        for i in range(self.dim_z):
            self.H[i, i] = 1

        R = OCSORTKalman._shared_R if OCSORTKalman._shared_R is not None else 0.05
        Q_pos = OCSORTKalman._shared_Q_pos if OCSORTKalman._shared_Q_pos is not None else 0.01
        Q_vel = OCSORTKalman._shared_Q_vel if OCSORTKalman._shared_Q_vel is not None else 0.0001

        self.R = np.eye(self.dim_z) * R
        self.P = np.eye(self.dim_x) * 10.0
        self.P[4:, 4:] *= 1000.0
        self.Q = np.eye(self.dim_x)
        self.Q[:4, :4] *= Q_pos
        self.Q[4:, 4:] *= Q_vel

    def initiate(self, measurement: np.ndarray) -> tuple:
        """根据初始观测框初始化状态均值与协方差矩阵.

        Args:
            measurement: 初始测量值 [cx, cy, a, h].

        Returns:
            初始状态均值向量与协方差矩阵.
        """
        mean = np.zeros(self.dim_x)
        mean[:self.dim_z] = measurement
        covariance = self.P.copy()
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """执行卡尔曼滤波器的一步前向状态预测.

        Args:
            mean: 当前状态均值向量.
            covariance: 当前状态协方差矩阵.

        Returns:
            预测后的状态均值与协方差矩阵.
        """
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + self.Q
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:
        """使用最新测量值执行卡尔曼更新步骤.

        Args:
            mean: 预测状态均值.
            covariance: 预测状态协方差.
            measurement: 实际观测值 [cx, cy, a, h].

        Returns:
            更新后的状态均值与协方差矩阵.
        """
        projected_mean = self.H @ mean
        projected_cov = self.H @ covariance @ self.H.T + self.R
        kalman_gain = covariance @ self.H.T @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean
        mean = mean + kalman_gain @ innovation
        covariance = covariance - kalman_gain @ self.H @ covariance
        return mean, covariance


class TrackState:
    """轨迹状态枚举常量类."""

    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class Tracklet:
    """OC-SORT / ByteTrack 跟踪轨迹内部实体类."""

    def __init__(self, tlwh: np.ndarray, score: float) -> None:
        """初始化轨迹实体，建立卡尔曼滤波器并写入首帧观测.

        Args:
            tlwh: 首帧观测框 [top, left, width, height].
            score: 首帧观测置信度.
        """
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.tracklet_len = 0
        self.state = TrackState.NEW

        self.kalman = OCSORTKalman()
        self.mean, self.covariance = self.kalman.initiate(self.tlwh_to_xyah(self._tlwh))

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
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """将 [top, left, width, height] 转换为中心坐标与宽高比格式 [cx, cy, aspect_ratio, height].

        Args:
            tlwh: 边界框坐标 [x, y, w, h].

        Returns:
            转换后的状态向量 [cx, cy, a, h].
        """
        xyah = np.asarray(tlwh).copy()
        xyah[:2] += xyah[2:] / 2
        xyah[2] /= xyah[3]
        return xyah

    @staticmethod
    def xyah_to_tlwh(xyah: np.ndarray) -> np.ndarray:
        """将中心坐标与宽高比格式 [cx, cy, aspect_ratio, height] 转换为 [top, left, width, height].

        Args:
            xyah: 状态向量 [cx, cy, a, h].

        Returns:
            转换后的边界框 [x, y, w, h].
        """
        tlwh = np.asarray(xyah).copy()
        tlwh[2] *= tlwh[3]
        tlwh[:2] -= tlwh[2:] / 2
        return tlwh

    @property
    def tlwh(self) -> np.ndarray:
        """获取当前估计的 [top_left_x, top_left_y, width, height] 边界框."""
        return self.xyah_to_tlwh(self.mean[:4])

    @property
    def tlbr(self) -> np.ndarray:
        """获取当前估计的 [x1, y1, x2, y2] 边界框."""
        tlbr = self.tlwh.copy()
        tlbr[2:] += tlbr[:2]
        return tlbr

    @property
    def bbox(self) -> np.ndarray:
        """获取标准包围盒坐标数组 [x1, y1, x2, y2]."""
        return self.tlbr

    def predict(self) -> None:
        """执行卡尔曼滤波前向预测步骤，更新内部状态均值与协方差."""
        self.mean, self.covariance = self.kalman.predict(self.mean, self.covariance)

    def update(self, tlwh: np.ndarray, score: float, frame_id: Optional[int] = None) -> None:
        """使用新的检测框观测更新轨迹状态与卡尔曼滤波器.

        Args:
            tlwh: 当前检测到的边界框 [x, y, w, h].
            score: 当前检测置信度得分.
            frame_id: 当前视频帧号，可选.
        """
        self._tlwh = tlwh
        self.score = score

        if frame_id is not None:
            self.frame_id = frame_id

        self.mean, self.covariance = self.kalman.update(
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

    def mark_lost(self) -> None:
        """将当前轨迹状态标记为丢失 (TrackState.LOST)."""
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        """将当前轨迹状态标记为移除 (TrackState.REMOVED)."""
        self.state = TrackState.REMOVED

    def get_center(self) -> np.ndarray:
        """获取轨迹当前的中心点坐标 [cx, cy]."""
        tlwh = self.tlwh
        return np.array([tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2])

    def get_velocity(self) -> np.ndarray:
        """获取轨迹当前的估计速度矢量 [vx, vy]."""
        if len(self.position_history) < 2:
            return np.zeros(2)
        window = min(5, len(self.position_history))
        positions = list(self.position_history)[-window:]
        if len(positions) < 2:
            return np.zeros(2)
        velocity = np.array(positions[-1]) - np.array(positions[0])
        velocity /= (len(positions) - 1)
        return velocity

    def oc_sort_recover(
        self,
        tlwh: np.ndarray,
        score: float,
        lost_frames: int,
        frame_id: Optional[int] = None,
    ) -> None:
        """执行 OC-SORT 观察恢复（Observation-Centric Recovery）.

        Args:
            tlwh: 新匹配到的观测框 [x, y, w, h].
            score: 新检测置信度.
            lost_frames: 轨迹已连续丢失的帧数.
            frame_id: 当前视频帧号，可选.
        """
        if self.last_observation is not None:
            old_center = self.last_observation[1]
            new_center = np.array([tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2])

            if lost_frames > 0:
                avg_velocity = (new_center - old_center) / lost_frames
                self.mean[4] = avg_velocity[0]
                self.mean[5] = avg_velocity[1]
                self.kalman.R = np.eye(4) * 0.1

        self.update(tlwh, score, frame_id=frame_id)
        self.state = TrackState.TRACKED


class OCSORTByteTracker:
    """OC-SORT + ByteTrack 融合跟踪器：多阶段匈牙利匹配维护人员轨迹."""

    def __init__(
        self,
        track_threshold: float = 0.50,
        match_threshold: float = 0.80,
        track_buffer: int = 30000,
        frame_rate: int = 30,
        min_center_distance: float = 150.0,
        kalman_R: float = 0.05,
        kalman_Q_pos: float = 0.01,
        kalman_Q_vel: float = 0.0001,
        confirm_frames: int = 3,
        max_recover_distance: float = 300.0,
        max_speed_per_frame: float = 10.0,
    ) -> None:
        """初始化跟踪超参数、卡尔曼共享噪声与四个轨迹池.

        Args:
            track_threshold: 创建新轨迹的最低检测置信度，默认 0.50.
            match_threshold: 阶段一 OC-SORT 匹配的最大允许代价，默认 0.80.
            track_buffer: 轨迹缓冲区帧数，默认 30000.
            frame_rate: 视频帧率，用于折算最长丢失时长，默认 30.
            min_center_distance: 新建轨迹时的最小中心距离（像素），默认 150.0.
            kalman_R: 卡尔曼测量噪声权重，默认 0.05.
            kalman_Q_pos: 卡尔曼位置过程噪声权重，默认 0.01.
            kalman_Q_vel: 卡尔曼速度过程噪声权重，默认 0.0001.
            confirm_frames: tentative 轨迹确认为正式轨迹所需的连续匹配帧数，默认 3.
            max_recover_distance: 恢复丢失轨迹的最大允许距离（像素），默认 300.0.
            max_speed_per_frame: 允许的单帧最大位移（像素），默认 10.0.
        """
        OCSORTKalman.set_parameters(kalman_R, kalman_Q_pos, kalman_Q_vel)

        self.track_threshold = track_threshold
        self.match_threshold = match_threshold
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.min_center_distance = min_center_distance
        self.confirm_frames = confirm_frames
        self.max_recover_distance = max_recover_distance
        self.max_speed_per_frame = max_speed_per_frame

        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

        self._center_distance_threshold = 0.6
        self._center_norm_threshold = 4.0

        self.tracked_tracks: List[Tracklet] = []
        self.lost_tracks: List[Tracklet] = []
        self.removed_tracks: List[Tracklet] = []
        self.tentative_tracks: List[Tracklet] = []

        self.frame_id = 0
        self.track_id = 0

    def _get_predicted_center(self, track: Tracklet) -> np.ndarray:
        """计算轨迹在当前帧的预测中心点位置.

        Args:
            track: 目标轨迹对象.

        Returns:
            预测中心坐标 [cx, cy].
        """
        if track.last_observation is None:
            return track.get_center()

        last_center = track.last_observation[1]
        elapsed = self.frame_id - track.last_observation[0]

        if elapsed <= 0:
            return last_center

        velocity = track.get_velocity()
        if np.linalg.norm(velocity) > 0:
            max_extrapolation = 15
            extrapolation_frames = min(elapsed, max_extrapolation)
            predicted = last_center + velocity * extrapolation_frames
            return predicted

        return last_center

    def update(
        self,
        high_detections: List[Dict],
        low_detections: Optional[List[Dict]] = None,
    ) -> List[Tracklet]:
        """执行单帧多阶段匹配，返回本帧已确认的轨迹列表.

        Args:
            high_detections: 高置信度检测结果列表.
            low_detections: 低置信度检测结果列表，缺省视为空.

        Returns:
            已激活（is_activated）的轨迹列表.
        """
        self.frame_id += 1

        if low_detections is None:
            low_detections = []

        activated_tracks = []
        refound_tracks = []
        newly_lost_tracks = []
        newly_removed_tracks = []

        high_measurements = self._convert_detections(high_detections)
        low_measurements = self._convert_detections(low_detections)

        for track in self.tracked_tracks:
            track.predict()
        for track in self.lost_tracks:
            track.predict()
        for track in self.tentative_tracks:
            track.predict()

        # ============================================================
        # 阶段一：confirmed tracked + tentative vs high_detections
        # ============================================================
        tracked_pool = [t for t in self.tracked_tracks if t.state == TrackState.TRACKED]
        tentative_pool = [t for t in self.tentative_tracks if t.state == TrackState.TRACKED]
        all_tracked_pool = tracked_pool + tentative_pool

        if len(all_tracked_pool) > 0 and len(high_measurements) > 0:
            tracked_cost_matrix = self._compute_oc_sort_distance(all_tracked_pool, high_measurements)

            for track_idx, track in enumerate(all_tracked_pool):
                if track.last_observation is None:
                    continue
                last_center = track.last_observation[1]
                for det_idx, det in enumerate(high_measurements):
                    det_center = np.array([det['tlwh'][0] + det['tlwh'][2] / 2,
                                           det['tlwh'][1] + det['tlwh'][3] / 2])
                    observation_distance = np.linalg.norm(det_center - last_center)
                    if observation_distance > self.max_recover_distance:
                        tracked_cost_matrix[track_idx, det_idx] = 1.0

            matches_tracked, unmatched_tracked_idx, unmatched_high_det_idx = self._linear_assignment(
                tracked_cost_matrix, threshold=self.match_threshold
            )
        else:
            matches_tracked = []
            unmatched_tracked_idx = list(range(len(all_tracked_pool)))
            unmatched_high_det_idx = list(range(len(high_measurements)))

        confirmed_activated = []
        tentative_matched = []

        for track_idx, det_idx in matches_tracked:
            track = all_tracked_pool[track_idx]
            det = high_measurements[det_idx]
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

        activated_tracks.extend(confirmed_activated)

        unmatched_tracked = [all_tracked_pool[track_idx] for track_idx in unmatched_tracked_idx]
        unmatched_confirmed = [t for t in unmatched_tracked if t.is_activated]
        unmatched_tentative = [t for t in unmatched_tracked if not t.is_activated]
        unmatched_high_measurements = [high_measurements[det_idx] for det_idx in unmatched_high_det_idx]

        # ============================================================
        # 阶段二：lost轨迹 vs 未匹配high_detections (中心距离+速度预测)
        # ============================================================
        unmatched_lost_tracks = []
        if len(self.lost_tracks) > 0 and len(unmatched_high_measurements) > 0:
            lost_cost_matrix = self._compute_center_distance(self.lost_tracks, unmatched_high_measurements)
            matches_lost, unmatched_lost_idx, unmatched_high_det_idx = self._linear_assignment(
                lost_cost_matrix, threshold=self._center_distance_threshold
            )

            for lost_idx, det_idx in matches_lost:
                track = self.lost_tracks[lost_idx]
                det = unmatched_high_measurements[det_idx]
                lost_frames = self.frame_id - track.last_observation[0] if track.last_observation else 0
                if lost_frames > 5:
                    track.oc_sort_recover(det['tlwh'], det['score'], lost_frames, frame_id=self.frame_id)
                else:
                    track.update(det['tlwh'], det['score'], frame_id=self.frame_id)
                refound_tracks.append(track)

            remaining_high_measurements = [unmatched_high_measurements[det_idx] for det_idx in unmatched_high_det_idx]
            unmatched_lost_tracks = [self.lost_tracks[lost_idx] for lost_idx in unmatched_lost_idx]
        else:
            remaining_high_measurements = unmatched_high_measurements
            unmatched_lost_tracks = list(self.lost_tracks)

        for track in unmatched_confirmed:
            track.mark_lost()
            newly_lost_tracks.append(track)

        for track in unmatched_tentative:
            track.mark_lost()

        # ============================================================
        # 阶段二.五：剩余lost轨迹 vs low_detections (中心距离+速度预测)
        # ============================================================
        remaining_low_measurements = low_measurements
        if len(unmatched_lost_tracks) > 0 and len(low_measurements) > 0:
            lost_low_cost_matrix = self._compute_center_distance(unmatched_lost_tracks, low_measurements)
            matches_lost_low, unmatched_lost_idx, unmatched_low_det_idx = self._linear_assignment(
                lost_low_cost_matrix, threshold=self._center_distance_threshold
            )

            for lost_idx, det_idx in matches_lost_low:
                track = unmatched_lost_tracks[lost_idx]
                det = low_measurements[det_idx]
                lost_frames = self.frame_id - track.last_observation[0] if track.last_observation else 0
                if lost_frames > 5:
                    track.oc_sort_recover(det['tlwh'], det['score'], lost_frames, frame_id=self.frame_id)
                else:
                    track.update(det['tlwh'], det['score'], frame_id=self.frame_id)
                refound_tracks.append(track)

            remaining_low_measurements = [low_measurements[det_idx] for det_idx in unmatched_low_det_idx]

        # ============================================================
        # 阶段三：ByteTrack级匹配 - 未匹配tracked vs 剩余low_detections
        # ============================================================
        if len(unmatched_confirmed) > 0 and len(remaining_low_measurements) > 0:
            confirmed_low_cost_matrix = self._compute_iou_distance(unmatched_confirmed, remaining_low_measurements)

            for track_idx, track in enumerate(unmatched_confirmed):
                if track.last_observation is None:
                    continue
                last_center = track.last_observation[1]
                elapsed = self.frame_id - track.last_observation[0]
                max_allowed_distance = min(
                    self.max_recover_distance,
                    self.max_speed_per_frame * max(elapsed, 1)
                )
                for det_idx, det in enumerate(remaining_low_measurements):
                    det_center = np.array([det['tlwh'][0] + det['tlwh'][2] / 2,
                                           det['tlwh'][1] + det['tlwh'][3] / 2])
                    observation_distance = np.linalg.norm(det_center - last_center)
                    predicted = self._get_predicted_center(track)
                    predicted_distance = np.linalg.norm(det_center - predicted)
                    if observation_distance > max_allowed_distance and predicted_distance > max_allowed_distance:
                        confirmed_low_cost_matrix[track_idx, det_idx] = 1.0

            matches_low, unmatched_tracked_idx, unmatched_low_det_idx = self._linear_assignment(
                confirmed_low_cost_matrix, threshold=0.5
            )

            for track_idx, det_idx in matches_low:
                track = unmatched_confirmed[track_idx]
                det = remaining_low_measurements[det_idx]
                track.update(det['tlwh'], det['score'], frame_id=self.frame_id)
                activated_tracks.append(track)

        # ============================================================
        # 阶段四：新轨迹初始化 - 先尝试恢复lost，再创建tentative
        # ============================================================
        still_lost_tracks = [
            t for t in self.lost_tracks
            if t.state == TrackState.LOST and t not in refound_tracks
        ]
        still_lost_tracks.extend([t for t in newly_lost_tracks if t.state == TrackState.LOST])

        for det in remaining_high_measurements:
            if det['score'] < self.track_threshold:
                continue

            det_center = np.array([det['tlwh'][0] + det['tlwh'][2] / 2,
                                   det['tlwh'][1] + det['tlwh'][3] / 2])

            recovered = self._try_recover_lost(det, det_center, still_lost_tracks)
            if recovered is not None:
                refound_tracks.append(recovered)
                still_lost_tracks = [t for t in still_lost_tracks if t.track_id != recovered.track_id]
                continue

            too_close = self._check_spatial_filter(det_center)
            if too_close:
                continue

            new_track = Tracklet(det['tlwh'], det['score'])
            new_track.frame_id = self.frame_id
            new_track.start_frame = self.frame_id
            new_track.state = TrackState.TRACKED
            new_track.tracklet_len = 1
            new_track.is_activated = False
            new_track.last_observation = (self.frame_id, det_center.copy(), det['tlwh'].copy())
            new_track.observation_ages.append(self.frame_id)
            self.tentative_tracks.append(new_track)

        # ============================================================
        # 阶段四.五：tentative轨迹确认
        # 连续匹配confirm_frames帧后分配track_id
        # ============================================================
        newly_confirmed = []
        remaining_tentative = []
        for track in self.tentative_tracks:
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

        activated_tracks.extend(newly_confirmed)
        self.tentative_tracks = remaining_tentative

        # 超时轨迹删除
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                newly_removed_tracks.append(track)

        # 更新轨迹列表
        tracked_ids = set()
        unique_tracked = []
        for t in self.tracked_tracks:
            if t.state == TrackState.TRACKED and t.is_activated and t.track_id not in tracked_ids:
                tracked_ids.add(t.track_id)
                unique_tracked.append(t)
        self.tracked_tracks = unique_tracked

        for t in activated_tracks:
            if t.is_activated and t.track_id not in tracked_ids:
                self.tracked_tracks.append(t)
                tracked_ids.add(t.track_id)

        for t in refound_tracks:
            if t.is_activated and t.track_id not in tracked_ids:
                self.tracked_tracks.append(t)
                tracked_ids.add(t.track_id)

        self.lost_tracks = [t for t in self.lost_tracks if t.state == TrackState.LOST]
        self.lost_tracks.extend(newly_lost_tracks)

        self.removed_tracks.extend(newly_removed_tracks)

        return [t for t in self.tracked_tracks if t.state == TrackState.TRACKED and t.is_activated]

    def _try_recover_lost(
        self,
        det: dict,
        det_center: np.ndarray,
        candidate_tracks: list,
    ) -> Optional[Tracklet]:
        """尝试将未匹配的检测框与已丢失的轨迹进行恢复匹配.

        Args:
            det: 检测框字典.
            det_center: 检测框中心点.
            candidate_tracks: 候选的已丢失轨迹列表.

        Returns:
            恢复成功的轨迹对象；未恢复时返回 None.
        """
        if len(candidate_tracks) == 0:
            return None

        best_track = None
        best_cost = float('inf')

        for track in candidate_tracks:
            if track.last_observation is None:
                continue

            last_center = track.last_observation[1]
            last_tlwh = track.last_observation[2]
            last_area = last_tlwh[2] * last_tlwh[3]
            elapsed = self.frame_id - track.last_observation[0]

            max_allowed_distance = min(
                self.max_recover_distance,
                self.max_speed_per_frame * max(elapsed, 1)
            )

            predicted_center = self._get_predicted_center(track)

            observation_distance = np.linalg.norm(det_center - last_center)
            predicted_distance = np.linalg.norm(det_center - predicted_center)

            if observation_distance > max_allowed_distance and predicted_distance > max_allowed_distance:
                continue

            center_distance = min(observation_distance, predicted_distance)

            det_area = det['tlwh'][2] * det['tlwh'][3]
            reference_size = max(np.sqrt(last_area), np.sqrt(det_area))
            if reference_size == 0:
                continue

            normalized_distance = center_distance / reference_size

            adaptive_threshold = self._center_norm_threshold * (1.0 + 0.01 * min(elapsed, 50))
            adaptive_threshold = min(adaptive_threshold, self._center_norm_threshold * 2.0)

            if normalized_distance < adaptive_threshold:
                cost = normalized_distance / adaptive_threshold
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

    def _check_spatial_filter(self, det_center: np.ndarray) -> bool:
        """检查新检测框是否与已有活跃轨迹保持最小空间距离过滤.

        Args:
            det_center: 检测框中心坐标.

        Returns:
            检测框与已有轨迹过近、需被过滤时返回 True.
        """
        all_active = self.tracked_tracks + self.lost_tracks + self.tentative_tracks
        for track in all_active:
            if track.state in [TrackState.TRACKED, TrackState.LOST]:
                if track.last_observation is not None:
                    reference_center = track.last_observation[1]
                    predicted = self._get_predicted_center(track)

                    observation_distance = np.linalg.norm(det_center - reference_center)
                    predicted_distance = np.linalg.norm(det_center - predicted)

                    min_distance = min(observation_distance, predicted_distance)
                else:
                    min_distance = np.linalg.norm(det_center - track.get_center())

                if min_distance < self.min_center_distance:
                    return True
        return False

    def _convert_detections(self, detections: List[Dict]) -> List[Dict]:
        """将输入检测字典列表转换为内部标准 [tlwh, score] 格式.

        Args:
            detections: 原始检测框列表.

        Returns:
            转换后的检测列表.
        """
        converted = []
        for det in detections:
            box = det['box']
            converted.append({
                'tlwh': np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]]),
                'score': det['confidence']
            })
        return converted

    def _compute_oc_sort_distance(self, tracks: list, detections: list) -> np.ndarray:
        """计算轨迹与检测框之间的 OC-SORT 综合距离代价矩阵.

        Args:
            tracks: 轨迹列表.
            detections: 检测框列表.

        Returns:
            代价矩阵 (N, M).
        """
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

        for track_idx, track in enumerate(tracks):
            for det_idx, det in enumerate(detections):
                iou = self._iou(track.tlbr, det['tlwh'])

                direction_bonus = 0.0
                if len(track.position_history) >= 2 and track.state == TrackState.TRACKED:
                    direction = track.get_velocity()
                    det_center = np.array([det['tlwh'][0] + det['tlwh'][2] / 2,
                                           det['tlwh'][1] + det['tlwh'][3] / 2])
                    track_center = track.get_center()
                    predicted_center = track_center + direction

                    predicted_error = np.linalg.norm(det_center - predicted_center)
                    box_diagonal = np.sqrt(det['tlwh'][2] ** 2 + det['tlwh'][3] ** 2)
                    if box_diagonal > 0:
                        direction_bonus = 0.2 * (1.0 - min(predicted_error / box_diagonal, 1.0))

                cost_matrix[track_idx, det_idx] = 1.0 - iou - direction_bonus

        return cost_matrix

    def _compute_center_distance(self, tracks: list, detections: list) -> np.ndarray:
        """计算轨迹与检测框之间的欧氏中心距离代价矩阵.

        Args:
            tracks: 轨迹列表.
            detections: 检测框列表.

        Returns:
            代价矩阵 (N, M).
        """
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        cost_matrix = np.ones((len(tracks), len(detections)), dtype=np.float32)

        for track_idx, track in enumerate(tracks):
            if track.last_observation is None:
                continue

            last_center = track.last_observation[1]
            last_tlwh = track.last_observation[2]
            last_area = last_tlwh[2] * last_tlwh[3]
            elapsed = self.frame_id - track.last_observation[0]

            predicted_center = self._get_predicted_center(track)

            max_allowed_distance = min(
                self.max_recover_distance,
                self.max_speed_per_frame * max(elapsed, 1)
            )

            for det_idx, det in enumerate(detections):
                det_center = np.array([det['tlwh'][0] + det['tlwh'][2] / 2,
                                       det['tlwh'][1] + det['tlwh'][3] / 2])
                det_area = det['tlwh'][2] * det['tlwh'][3]

                observation_distance = np.linalg.norm(det_center - last_center)
                predicted_distance = np.linalg.norm(det_center - predicted_center)

                if observation_distance > max_allowed_distance and predicted_distance > max_allowed_distance:
                    continue

                center_distance = min(observation_distance, predicted_distance)

                reference_size = max(np.sqrt(last_area), np.sqrt(det_area))
                if reference_size == 0:
                    continue

                normalized_distance = center_distance / reference_size

                adaptive_threshold = self._center_norm_threshold * (1.0 + 0.01 * min(elapsed, 50))
                adaptive_threshold = min(adaptive_threshold, self._center_norm_threshold * 2.0)

                if normalized_distance < adaptive_threshold:
                    cost_matrix[track_idx, det_idx] = normalized_distance / adaptive_threshold

        return cost_matrix

    def _compute_iou_distance(self, tracks: list, detections: list) -> np.ndarray:
        """计算轨迹与检测框之间的纯 IoU 距离代价矩阵 (1 - IoU).

        Args:
            tracks: 轨迹列表.
            detections: 检测框列表.

        Returns:
            代价矩阵 (N, M).
        """
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

        for track_idx, track in enumerate(tracks):
            for det_idx, det in enumerate(detections):
                iou = self._iou(track.tlbr, det['tlwh'])
                cost_matrix[track_idx, det_idx] = 1.0 - iou

        return cost_matrix

    @staticmethod
    def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个矩形框 [tlwh] 之间的交并比 (IoU).

        Args:
            box1: 矩形框 1 [x, y, w, h].
            box2: 矩形框 2 [x, y, w, h].

        Returns:
            交并比数值 (0.0 ~ 1.0).
        """
        if box2.shape[0] == 4:
            box2 = np.array([box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]])

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    @staticmethod
    def _linear_assignment(cost_matrix: np.ndarray, threshold: float) -> tuple:
        """使用匈牙利算法求解二分图最小代价匹配.

        Args:
            cost_matrix: 代价矩阵 (N, M).
            threshold: 最大允许匹配代价阈值.

        Returns:
            匹配对、未匹配行、未匹配列.
        """
        if cost_matrix.size == 0:
            if cost_matrix.ndim == 0:
                return [], [], []
            elif cost_matrix.ndim == 1:
                return [], list(range(cost_matrix.shape[0])), []
            else:
                unmatched_rows = list(range(cost_matrix.shape[0])) if cost_matrix.shape[0] > 0 else []
                unmatched_cols = list(range(cost_matrix.shape[1])) if cost_matrix.shape[1] > 0 else []
                return [], unmatched_rows, unmatched_cols

        matches = []
        unmatched_rows = list(range(cost_matrix.shape[0]))
        unmatched_cols = list(range(cost_matrix.shape[1]))

        if cost_matrix.shape[0] > 0 and cost_matrix.shape[1] > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] <= threshold:
                    matches.append((row, col))
                    if row in unmatched_rows:
                        unmatched_rows.remove(row)
                    if col in unmatched_cols:
                        unmatched_cols.remove(col)

        return matches, unmatched_rows, unmatched_cols
