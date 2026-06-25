"""
OC-SORT + ByteTrack 动态目标跟踪器

使用 SRC_v6.1 的自研跟踪算法，保留角色分配功能。
"""
import logging
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger("module.mot.tracker")

# 工位坐标
WORKSTATIONS = {
    "LEADER": (1049, 398),
    "ROAD1": (1563, 494),
    "ROAD2": (1146, 662),
}


class STrack:
    """单个跟踪轨迹适配器"""

    def __init__(self, track_id: int, bbox: np.ndarray, score: float):
        self.track_id = int(track_id)
        self.bbox = np.array(bbox)  # [x1, y1, x2, y2]
        self.score = float(score)
        self.role: Optional[str] = None

    def get_center(self) -> np.ndarray:
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        ])


class MultiObjectTracker:
    """
    OC-SORT + ByteTrack 跟踪器

    使用 SRC_v6.1 的自研算法
    """

    def __init__(self):
        from modules.mot.ocsort_bytetrack import OCSORTByteTracker

        self.tracker = OCSORTByteTracker(
            track_thresh=0.50,
            match_thresh=0.80,
            track_buffer=30000,
            frame_rate=30,
            min_center_distance=150.0,
            kalman_R=0.05,
            kalman_Q_pos=0.01,
            kalman_Q_vel=0.0001,
            confirm_frames=3,
            max_recover_distance=300.0,
            max_speed_per_frame=10.0,
        )
        logger.info("跟踪器初始化完成")

        self.frame_id = 0
        self.roles_assigned = False
        self.role_map: Dict[str, int] = {}
        self._track_map: Dict[int, STrack] = {}
        self.initialized = False

    def assign_roles(self, tracks: List[STrack]) -> None:
        if self.roles_assigned or len(tracks) < 2:
            return

        ws_list = list(WORKSTATIONS.items())
        used_tracks = set()

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
        logger.info(f"角色分配完成: {self.role_map}")

    def track(self, frame: np.ndarray, detections: List[Dict]) -> List[STrack]:
        self.frame_id += 1
        self.initialized = True

        # 分离高低置信度检测
        high_dets = [d for d in detections if d['confidence'] >= 0.5]
        low_dets = [d for d in detections if d['confidence'] < 0.5]

        # 执行跟踪
        raw_tracks = self.tracker.update(high_dets, low_dets)

        # 转换
        tracks = []
        self._track_map.clear()
        for t in raw_tracks:
            track_id = t.track_id
            bbox = t.bbox
            score = t.score

            role = None
            for r, tid in self.role_map.items():
                if tid == track_id:
                    role = r
                    break

            st = STrack(track_id, bbox, score)
            st.role = role
            tracks.append(st)
            self._track_map[track_id] = st

        return tracks

    def get_track_by_role(self, role: str) -> Optional[STrack]:
        track_id = self.role_map.get(role)
        if track_id is None:
            return None
        return self._track_map.get(track_id)

    def reset(self):
        self.frame_id = 0
        self.roles_assigned = False
        self.role_map.clear()
        self._track_map.clear()
        self.initialized = False
