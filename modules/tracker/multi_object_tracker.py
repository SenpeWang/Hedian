"""多目标跟踪器：基于 OC-SORT 与 ByteTrack 算法实现人员跟踪与工位身份分配."""
import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger("module.tracker.tracker")

# 合法工位身份与基准坐标定义
WORKSTATIONS: Dict[str, tuple] = {
    "LEADER": (1049, 398),
    "ROAD1": (1563, 494),
    "ROAD2": (1146, 662),
}
VALID_IDENTITIES: Set[str] = {"LEADER", "ROAD1", "ROAD2"}


class STrack:
    """单个跟踪目标轨迹实体类.

    封装单个人员的全局跟踪 ID、分配身份、边界框坐标与置信度。
    """

    def __init__(
        self,
        track_id: int,
        bbox: np.ndarray,
        score: float,
        identity: Optional[str] = None,
    ):
        """初始化跟踪目标实体.

        Args:
            track_id (int): 全局唯一跟踪目标 ID.
            bbox (np.ndarray): 边界框坐标 [x_min, y_min, x_max, y_max].
            score (float): 目标检测或跟踪置信度得分.
            identity (Optional[str]): 人员身份，仅限 'LEADER'|'ROAD1'|'ROAD2'|None.
        """
        self.track_id: int = int(track_id)
        self.bbox: np.ndarray = np.array(bbox, dtype=float)
        self.score: float = float(score)
        self._identity: Optional[str] = (
            identity if identity in VALID_IDENTITIES else None
        )

    @property
    def identity(self) -> Optional[str]:
        """获取目标人员身份 (LEADER | ROAD1 | ROAD2 | None)."""
        return self._identity

    @identity.setter
    def identity(self, identity_value: Optional[str]) -> None:
        """设置目标人员身份，非法值自动置为 None.

        Args:
            identity_value (Optional[str]): 待设置的身份字符串.
        """
        self._identity = (
            identity_value if identity_value in VALID_IDENTITIES else None
        )

    def get_center(self) -> np.ndarray:
        """计算并获取目标中心点坐标 [center_x, center_y].

        Returns:
            np.ndarray: 中心点坐标一维数组 (2,).
        """
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        ])

    def to_dict(self) -> Dict[str, Any]:
        """将实体序列化为标准化字典结构.

        Returns:
            Dict[str, Any]: 包含 track_id, identity, bbox, score, center 的字典.
        """
        center_x, center_y = self.get_center()
        return {
            "track_id": self.track_id,
            "identity": self.identity,
            "bbox": [round(float(coord), 1) for coord in self.bbox],
            "score": round(self.score, 3),
            "center": [round(float(center_x), 1), round(float(center_y), 1)],
        }


class MultiObjectTracker:
    """多目标跟踪器.

    结合 OC-SORT 与 ByteTrack 双阈值匹配算法，实现连续帧目标跟踪与初始工位身份分配。
    """

    def __init__(self):
        """初始化多目标跟踪器."""
        from modules.tracker.ocsort_bytetrack import OCSORTByteTracker

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
        self.frame_id: int = 0
        self.identities_assigned: bool = False
        self.identity_map: Dict[str, int] = {}
        self._track_map: Dict[int, STrack] = {}
        self.initialized: bool = False
        logger.info("多目标跟踪器初始化完成")

    def assign_identities(self, tracks: List[STrack]) -> None:
        """根据工位基准坐标为跟踪目标分配初始身份 (LEADER, ROAD1, ROAD2).

        Args:
            tracks (List[STrack]): 当前帧中的跟踪目标列表.
        """
        if self.identities_assigned or len(tracks) < 2:
            return

        workstations_list = list(WORKSTATIONS.items())
        assigned_track_ids: set = set()

        assignments = []
        for identity_name, (workstation_x, workstation_y) in workstations_list:
            best_track, best_distance = None, float("inf")
            for track in tracks:
                if id(track) in assigned_track_ids:
                    continue
                center_x, center_y = track.get_center()
                distance = float(
                    np.linalg.norm(
                        np.array([center_x, center_y])
                        - np.array([workstation_x, workstation_y])
                    )
                )
                if distance < best_distance:
                    best_distance, best_track = distance, track
            if best_track is not None:
                assignments.append((identity_name, best_track))
                assigned_track_ids.add(id(best_track))

        for identity_name, track in assignments:
            track.identity = identity_name
            self.identity_map[identity_name] = track.track_id
            self._track_map[track.track_id] = track

        self.identities_assigned = True
        logger.info(f"工位身份分配完成: {self.identity_map}")

    def track(
        self, frame: np.ndarray, detections: List[Dict[str, Any]]
    ) -> List[STrack]:
        """处理单帧检测结果并更新人员跟踪轨迹.

        Args:
            frame (np.ndarray): 当前视频帧.
            detections (List[Dict[str, Any]]): 目标检测结果字典列表.

        Returns:
            List[STrack]: 更新后的 STrack 轨迹实体列表.
        """
        self.frame_id += 1
        self.initialized = True

        # 分离高低置信度检测（0.5 为 ByteTrack 双阈值分界）
        high_conf_detections = [
            detection for detection in detections if detection["confidence"] >= 0.5
        ]
        low_conf_detections = [
            detection for detection in detections if detection["confidence"] < 0.5
        ]

        raw_tracks = self.tracker.update(high_conf_detections, low_conf_detections)

        tracks: List[STrack] = []
        self._track_map.clear()
        for raw_track in raw_tracks:
            track_id = int(raw_track.track_id)
            bbox_array = raw_track.bbox
            score_val = float(raw_track.score)

            # 查找已分配的身份
            assigned_identity = None
            for id_name, mapped_tid in self.identity_map.items():
                if mapped_tid == track_id:
                    assigned_identity = id_name
                    break

            track_obj = STrack(
                track_id=track_id,
                bbox=bbox_array,
                score=score_val,
                identity=assigned_identity,
            )
            tracks.append(track_obj)
            self._track_map[track_id] = track_obj

        return tracks

    def get_track_by_identity(self, identity: str) -> Optional[STrack]:
        """根据身份名称获取对应的跟踪实体.

        Args:
            identity (str): 身份名称 ('LEADER' | 'ROAD1' | 'ROAD2').

        Returns:
            Optional[STrack]: 匹配的 STrack 对象；不存在或未分配时返回 None.
        """
        track_id = self.identity_map.get(identity)
        if track_id is None:
            return None
        return self._track_map.get(track_id)
