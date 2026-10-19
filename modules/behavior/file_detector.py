"""手指指向文件检测模块.

基于指向手框被 file 框包含的比例判定，采用合并式去重（同角色间隔不超过
merge_sec 视为同一次指向）+ 遮挡宽限（grace_frames）填补短暂漏检。
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from modules.behavior.base_detector import BaseDetector
from modules.behavior.behavior_utils import (
    PoseRoleClassifier,
    bbox_containment_ratio,
)
from modules.behavior.behavior_vis import draw_trigger

logger = logging.getLogger("module.behavior.file_detector")


class FingerFileDetector(BaseDetector):
    """手指指向文件行为检测器.

    使用指向手框与 file 框的包含比进行判定，同一动作只记录一条事件。
    """

    def __init__(
        self,
        file_contain_threshold: float = 0.7,
        merge_sec: float = 5.0,
        role_cooldown_sec: float = 1.5,
        grace_frames: int = 12,
        fps: float = 30.0,
    ) -> None:
        """初始化手指指向文件检测器.

        Args:
            file_contain_threshold: 手框被文件框包含的面积比阈值.
            merge_sec: 同角色事件的合并窗口（秒），持续指向只记一次.
            role_cooldown_sec: 角色抖动冷却（秒），避免同一动作切成多段.
            grace_frames: 遮挡宽限帧数，短暂漏检仍视为持续指向.
            fps: 视频采样帧率.
        """
        super().__init__(role_cooldown_sec, fps)
        self.file_contain_threshold: float = file_contain_threshold
        self.merge_sec: float = merge_sec
        self.role_cooldown_sec: float = role_cooldown_sec
        self.grace_frames: int = grace_frames
        # 最近一次判定为指向文件的帧号与角色（用于遮挡宽限延续）
        self._last_pointing_frame: int = -10_000
        self._last_pointing_role: Optional[str] = None

    def reset(self) -> None:
        """重置合并去重与遮挡宽限状态（新一轮推理解绑旧状态）."""
        self._last_event_frame = -10_000
        self._last_merged_frame = -10_000
        self._last_merged_role = None
        self._last_pointing_frame = -10_000
        self._last_pointing_role = None

    def detect(
        self,
        frame: np.ndarray,
        pointing_bboxes: Sequence[Sequence[float]],
        file_bboxes: Sequence[Sequence[float]],
        frame_count: int,
        role_classifier: Optional[PoseRoleClassifier] = None,
        role_frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """检测当前帧中是否存在手指向文件的行为.

        Args:
            frame: 当前视频帧（原地绘制触发框）.
            pointing_bboxes: 本帧判定为指向的手框列表.
            file_bboxes: 本帧检出的文件框列表.
            frame_count: 当前视频帧号.
            role_classifier: 角色判别器；为 None 时事件角色为空.
            role_frame: 供角色判别的干净帧（未叠加标注）；
                为 None 时退回使用 frame.

        Returns:
            触发事件字典列表，包含 event / state / role / contain_ratio 等字段.
        """
        events: List[Dict[str, Any]] = []
        best_contain_ratio = 0.0
        file_bboxes_hit: List[Sequence[float]] = []

        for hand_bbox in pointing_bboxes:
            contain_ratio = 0.0
            for file_bbox in file_bboxes:
                contain_ratio = max(
                    contain_ratio,
                    bbox_containment_ratio(hand_bbox, file_bbox),
                )
            best_contain_ratio = max(best_contain_ratio, contain_ratio)
            if contain_ratio >= self.file_contain_threshold:
                file_bboxes_hit.append(hand_bbox)

        if not file_bboxes_hit:
            # 遮挡宽限：短暂漏检仍视为持续指向，顺延合并窗口避免切成多段
            within_grace = (
                frame_count - self._last_pointing_frame
            ) <= self.grace_frames
            if within_grace and self._last_pointing_role is not None:
                self._merge_ready(
                    self._last_pointing_role,
                    frame_count,
                    self.merge_sec,
                    self.role_cooldown_sec,
                )
            return events

        role: Optional[str] = None
        if role_classifier is not None:
            role = role_classifier.classify(
                frame if role_frame is None else role_frame,
                file_bboxes_hit,
            )
        self._last_pointing_frame = frame_count
        self._last_pointing_role = role

        if not self._merge_ready(
            role, frame_count, self.merge_sec, self.role_cooldown_sec
        ):
            return events

        draw_trigger(frame, file_bboxes_hit, "FINGER_FILE!")
        events.append(
            self._make_event(
                "FINGER_FILE",
                "手指指向文件",
                frame_count,
                {"contain_ratio": round(float(best_contain_ratio), 3)},
                role=role,
            )
        )
        return events
