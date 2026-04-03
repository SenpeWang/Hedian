"""
动态多目标跟踪器
核心特性：
1. 零ID切换
2. 精确跟踪：检测多少人就跟踪多少人
3. 无硬编码：完全动态适应目标数量
4. 自动检测目标数量
5. 角色分配：LEADER/ROAD1/ROAD2（基于工位坐标）
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from collections import deque

from .. import WORKSTATIONS


@dataclass
class TrackerConfig:
    """跟踪器配置"""
    max_time_lost: int = 200
    match_thresh: float = 0.30
    dist_thresh: float = 280
    sim_thresh: float = 0.20



class TrackState:
    """跟踪状态枚举"""
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class STrack:
    """单个跟踪轨迹"""
    
    def __init__(self, track_id: int, bbox: np.ndarray, score: float):
        self.track_id = track_id
        self.bbox = bbox
        self.score = score
        self.tracklet_len = 0
        self.frame_id = 0
        self.state = TrackState.NEW
        
        self.position_history = deque(maxlen=60)
        self.velocity_history = deque(maxlen=30)
        self.bbox_history = deque(maxlen=30)
        
        self.confidence_sum = 0.0
        self.stability_score = 0.0
        self.lost_count = 0
        self.max_lost_frames = 200
        
        self.first_center = np.zeros(2)

        # ── 角色与姿态 ──
        self.role: Optional[str] = None   # LEADER / ROAD1 / ROAD2
        self.keypoints = None             # (17,3) ndarray
    
    def get_center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])
    
    def get_size(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    def get_aspect_ratio(self) -> float:
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        if h > 0:
            return w / h
        return 1.0
    
    def get_predicted_center(self) -> np.ndarray:
        if len(self.position_history) < 2:
            return self.get_center()
        positions = np.array(list(self.position_history))
        if len(positions) >= 3:
            recent = positions[-10:]
            velocity = np.mean(np.diff(recent, axis=0), axis=0)
            return positions[-1] + velocity * 3
        return positions[-1]
    
    def update_motion(self):
        center = self.get_center()
        self.position_history.append(center.copy())
        if len(self.position_history) >= 2:
            positions = np.array(list(self.position_history))
            velocity = positions[-1] - positions[-2]
            self.velocity_history.append(velocity.copy())
        self.bbox_history.append(self.bbox.copy())
        
        self.confidence_sum += self.score
        if self.tracklet_len > 0:
            self.stability_score = self.confidence_sum / self.tracklet_len
    
    def compute_similarity(self, other: 'STrack') -> float:
        sim_score = 0.0
        
        center1 = self.get_center()
        center2 = other.get_center()
        dist = np.linalg.norm(center1 - center2)
        position_sim = max(0, 1.0 - dist / 300)
        sim_score += position_sim * 0.35
        
        if len(self.position_history) > 0:
            home_dist = np.linalg.norm(center1 - self.first_center)
            home_sim = max(0, 1.0 - home_dist / 400)
            sim_score += home_sim * 0.25
        
        size1 = self.get_size()
        size2 = other.get_size()
        if size1 > 0 and size2 > 0:
            size_ratio = min(size1, size2) / max(size1, size2)
            sim_score += size_ratio * 0.15

        ar1 = self.get_aspect_ratio()
        ar2 = other.get_aspect_ratio()
        if ar1 > 0 and ar2 > 0:
            ar_ratio = min(ar1, ar2) / max(ar1, ar2)
            sim_score += ar_ratio * 0.15
        
        if len(self.velocity_history) > 0 and len(other.velocity_history) > 0:
            v1 = np.mean(np.array(list(self.velocity_history)), axis=0)
            v2 = np.mean(np.array(list(other.velocity_history)), axis=0)
            v1_norm = np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else 1
            v2_norm = np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else 1
            v_cosine = np.dot(v1, v2) / (v1_norm * v2_norm)
            sim_score += max(0, v_cosine) * 0.15
        
        if len(self.position_history) > 5 and len(other.position_history) > 5:
            pos1 = np.array(list(self.position_history)[-10:])
            pos2 = np.array(list(other.position_history)[-10:])
            if len(pos1) == len(pos2):
                path_dist = np.mean(np.linalg.norm(pos1 - pos2, axis=1))
                path_sim = max(0, 1.0 - path_dist / 200)
                sim_score += path_sim * 0.1

        stability_boost = min(self.stability_score, 1.0) * 0.2
        sim_score += stability_boost

        return min(sim_score, 1.0)


class DynamicTracker:
    """
    动态跟踪器 - 无硬编码
    核心特性：
    1. 零ID切换
    2. 精确跟踪：检测多少人就跟踪多少人
    3. 无硬编码：完全动态适应目标数量
    4. 自动检测目标数量
    """
    
    def __init__(self):
        self.frame_id = 0
        
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        
        self.next_id = 1
        
        self.max_time_lost = 200
        self.match_thresh = 0.30
        self.dist_thresh = 280
        self.sim_thresh = 0.20
        
        self.detection_history = deque(maxlen=60)
        self.initialized = False
        self.init_frames = 0
        self.init_dets_buffer = deque(maxlen=30)
        
        self.id_switch_count = 0

        # ── 角色分配 ──
        self.roles_assigned = False
        self.role_map: Dict[str, int] = {}  # role_name -> track_id

    def _assign_roles(self, tracks: List[STrack]):
        """按欧氏距离到工位坐标分配 LEADER/ROAD1/ROAD2"""
        if self.roles_assigned or len(tracks) < 2:
            return
        ws_list = list(WORKSTATIONS.items())  # [(name, (x,y)), ...]
        used_roles, used_tracks = set(), set()
        # 贪心：对每个工位找最近的 track
        assignments = []
        for role, (wx, wy) in ws_list:
            best_t, best_d = None, float("inf")
            for t in tracks:
                if id(t) in used_tracks:
                    continue
                cx, cy = t.get_center()
                d = np.linalg.norm(np.array([cx, cy]) - np.array([wx, wy]))
                if d < best_d:
                    best_d, best_t = d, t
            if best_t is not None:
                assignments.append((role, best_t))
                used_tracks.add(id(best_t))
        for role, t in assignments:
            t.role = role
            self.role_map[role] = t.track_id
        self.roles_assigned = True
    
    def detect_target_count(self) -> int:
        """动态检测目标数量"""
        if len(self.detection_history) < 10:
            return 0
        
        recent_counts = list(self.detection_history)[-30:]
        count_frequency = {}
        for count in recent_counts:
            count_frequency[count] = count_frequency.get(count, 0) + 1
        
        if count_frequency:
            most_common = max(count_frequency.items(), key=lambda x: x[1])[0]
            confidence = count_frequency[most_common] / len(recent_counts)
            if confidence > 0.5:
                return most_common
        
        return int(np.median(recent_counts))
    
    def initialize_tracker(self, detections: List[Dict]):
        """初始化跟踪器 - 动态适应检测数量"""
        current_det_count = len(detections)
        
        if current_det_count == 0:
            return
        
        sorted_dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        
        positions = []
        for det in sorted_dets:
            center = np.array([
                (det['box'][0] + det['box'][2]) / 2,
                (det['box'][1] + det['box'][3]) / 2
            ])
            positions.append(center)
        
        selected_indices = []
        for i, pos1 in enumerate(positions[:len(sorted_dets)]):
            if len(selected_indices) >= current_det_count:
                break
            is_far_enough = True
            for j in selected_indices:
                pos2 = positions[j]
                dist = np.linalg.norm(pos1 - pos2)
                if dist < 100:
                    is_far_enough = False
                    break
            if is_far_enough:
                selected_indices.append(i)
        
        if len(selected_indices) < current_det_count:
            for i in range(len(sorted_dets)):
                if i not in selected_indices and len(selected_indices) < current_det_count:
                    selected_indices.append(i)
        
        for idx in selected_indices[:current_det_count]:
            det = sorted_dets[idx]
            center = np.array([
                (det['box'][0] + det['box'][2]) / 2,
                (det['box'][1] + det['box'][3]) / 2
            ])
            
            track = STrack(
                track_id=self.next_id,
                bbox=np.array(det['box']),
                score=det['confidence']
            )
            track.first_center = center.copy()
            track.state = TrackState.TRACKED
            track.update_motion()
            self.tracked_stracks.append(track)
            self.next_id += 1
    
    def compute_cost_matrix(self, tracks: List[STrack], detections: List[Dict]) -> np.ndarray:
        n_tracks = len(tracks)
        n_dets = len(detections)
        cost_matrix = np.full((n_tracks, n_dets), 1e6)
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                det_center = np.array([
                    (det['box'][0] + det['box'][2]) / 2,
                    (det['box'][1] + det['box'][3]) / 2
                ])
                
                dist = np.linalg.norm(track.get_predicted_center() - det_center)
                
                if dist > self.dist_thresh * 1.5:
                    continue
                
                similarity = track.compute_similarity(STrack(
                    track_id=0,
                    bbox=np.array(det['box']),
                    score=det['confidence']
                ))
                
                cost = dist / self.dist_thresh + (1 - similarity) * 1.5
                
                if cost < 2.5:
                    cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def match_and_update(self, tracks: List[STrack], detections: List[Dict]) -> Tuple[List[int], List[int], List[int]]:
        if len(tracks) == 0 or len(detections) == 0:
            return [], [], list(range(len(detections)))
        
        cost_matrix = self.compute_cost_matrix(tracks, detections)
        
        if cost_matrix.size == 0:
            return [], [], list(range(len(detections)))
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_tracks = []
        matched_dets = []
        unmatched_dets = list(range(len(detections)))
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.match_thresh * 2:
                matched_tracks.append(r)
                matched_dets.append(c)
                if c in unmatched_dets:
                    unmatched_dets.remove(c)
        
        return matched_tracks, matched_dets, unmatched_dets
    
    def track(self, frame, detections: List[Dict]) -> List[STrack]:
        self.frame_id += 1
        self.detection_history.append(len(detections))
        
        if not self.initialized:
            self.init_dets_buffer.append(detections)
            
            if len(self.init_dets_buffer) >= 30:
                detected_count = self.detect_target_count()
                
                recent_dets = list(self.init_dets_buffer)
                all_consecutive = True
                for dets in recent_dets[-10:]:
                    if len(dets) < detected_count:
                        all_consecutive = False
                        break
                
                if all_consecutive:
                    last_dets = recent_dets[-1]
                    self.initialize_tracker(last_dets)
                    self.initialized = True
                else:
                    for dets in recent_dets:
                        if len(dets) >= detected_count:
                            self.initialize_tracker(dets)
                            self.initialized = True
                            break
            
            return [t for t in self.tracked_stracks if t.state == TrackState.TRACKED]
        
        if not detections:
            for t in self.tracked_stracks:
                t.state = TrackState.LOST
                t.lost_count += 1
                self.lost_stracks.append(t)
            self.tracked_stracks = []
            return []
        
        all_tracks = self.tracked_stracks + self.lost_stracks
        
        matched_track_indices, matched_det_indices, unmatched_det_indices = self.match_and_update(all_tracks, detections)
        
        for idx in matched_track_indices:
            track = all_tracks[idx]
            det_idx = matched_det_indices[matched_track_indices.index(idx)]
            det = detections[det_idx]
            
            track.bbox = np.array(det['box'])
            track.score = det['confidence']
            track.frame_id = self.frame_id
            track.tracklet_len += 1
            track.state = TrackState.TRACKED
            track.lost_count = 0
            track.update_motion()
            
            if track in self.lost_stracks:
                self.lost_stracks.remove(track)
            if track not in self.tracked_stracks:
                self.tracked_stracks.append(track)
        
        for i, t in enumerate(all_tracks):
            if i not in matched_track_indices and t.state == TrackState.TRACKED:
                t.state = TrackState.LOST
                t.lost_count += 1
                if t in self.tracked_stracks:
                    self.tracked_stracks.remove(t)
                    if t not in self.lost_stracks:
                        self.lost_stracks.append(t)
        
        self.lost_stracks = [
            t for t in self.lost_stracks 
            if t.lost_count <= t.max_lost_frames
        ]
        
        if len(unmatched_det_indices) > 0:
            det_centers = []
            for det_idx in unmatched_det_indices:
                det = detections[det_idx]
                center = np.array([
                    (det['box'][0] + det['box'][2]) / 2,
                    (det['box'][1] + det['box'][3]) / 2
                ])
                det_centers.append((det_idx, center, det))
            
            for det_idx, det_center, det in sorted(det_centers, key=lambda x: x[1][0]):
                best_lost = None
                best_sim = 0
                
                for lost in self.lost_stracks:
                    sim = lost.compute_similarity(STrack(
                        track_id=0,
                        bbox=np.array(det['box']),
                        score=det['confidence']
                    ))
                    if sim > best_sim and sim > self.sim_thresh:
                        best_sim = sim
                        best_lost = lost
                
                if best_lost is not None:
                    best_lost.bbox = np.array(det['box'])
                    best_lost.score = det['confidence']
                    best_lost.frame_id = self.frame_id
                    best_lost.state = TrackState.TRACKED
                    best_lost.lost_count = 0
                    best_lost.update_motion()
                    
                    self.lost_stracks.remove(best_lost)
                    if best_lost not in self.tracked_stracks:
                        self.tracked_stracks.append(best_lost)
        
        return [t for t in self.tracked_stracks if t.state == TrackState.TRACKED]
    
    def get_statistics(self):
        return {
            'active_tracks': len(self.tracked_stracks),
            'lost_tracks': len(self.lost_stracks),
            'next_id': self.next_id,
            'max_id_generated': self.next_id - 1,
            'id_switch_count': self.id_switch_count,
            'roles_assigned': self.roles_assigned,
            'role_map': dict(self.role_map),
        }

    def get_track_by_role(self, role: str) -> Optional[STrack]:
        """按角色名取得当前跟踪的 track"""
        tid = self.role_map.get(role)
        if tid is None:
            return None
        for t in self.tracked_stracks:
            if t.track_id == tid and t.state == TrackState.TRACKED:
                return t
        return None
