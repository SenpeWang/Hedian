"""手指指向屏幕检测模块.

基于指向手框与预标注屏幕/机箱多边形区域的重叠比及最短距离判定，
采用上升沿触发（持续指向只记一次）+ debounce 防抖 + 冷却兜底。
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.behavior.base_detector import BaseDetector
from modules.behavior.behavior_utils import (
    PoseRoleClassifier,
    bbox_polygon_overlap_ratio,
    bbox_to_polygon_distance,
)
from modules.behavior.behavior_vis import (
    draw_roi_overlay,
    draw_trigger,
    screen_polygons_as_tuples,
)

logger = logging.getLogger("module.behavior.screen_detect")


class FingerScreenDetector(BaseDetector):
    """手指指向屏幕行为检测器.

    使用指向手框与机箱/屏幕 ROI 多边形的空间重叠度与几何距离进行判定。
    """

    def __init__(
        self,
        screen_overlap_threshold: float = 0.2,
        max_distance: int = 10,
        cooldown_sec: float = 1.5,
        fps: float = 30.0,
        debounce_frames: int = 3,
    ) -> None:
        """初始化手指指向屏幕检测器.

        Args:
            screen_overlap_threshold: 指向手与机箱区域面积重叠比阈值.
            max_distance: 指向手与机箱多边形边缘的最大允许像素距离.
            cooldown_sec: 动作触发后的冷却时间（秒）.
            fps: 视频采样帧率.
            debounce_frames: 连续多少帧无指向才算动作结束.
        """
        super().__init__(cooldown_sec, fps)
        self.screen_overlap_threshold: float = screen_overlap_threshold
        self.max_distance: int = max_distance
        self.debounce_frames: int = debounce_frames
        # 上升沿状态：当前是否处于持续指向中，以及连续无指向帧数
        self._active: bool = False
        self._idle_frames: int = 0

    @staticmethod
    def draw_roi(frame: np.ndarray) -> None:
        """在视频帧上绘制屏幕/机箱兴趣区域 (ROI) 多边形覆盖层.

        Args:
            frame: 原始视频帧 (BGR 格式).
        """
        draw_roi_overlay(frame, screen_polygons_as_tuples())

    def reset(self) -> None:
        """重置上升沿与冷却状态（新一轮推理解绑旧状态）."""
        self._active = False
        self._idle_frames = 0
        self._last_event_frame = -10_000

    def detect(
        self,
        frame: np.ndarray,
        pointing_bboxes: Sequence[Sequence[float]],
        screen_polygons: Sequence[Tuple[str, np.ndarray]],
        frame_count: int,
        role_classifier: Optional[PoseRoleClassifier] = None,
        role_frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """检测当前帧中是否存在手指向屏幕的行为.

        Args:
            frame: 当前视频帧（原地绘制触发框）.
            pointing_bboxes: 本帧判定为指向的手框列表.
            screen_polygons: (标签, 机箱多边形轮廓) 列表，已按帧尺寸缩放.
            frame_count: 当前视频帧号.
            role_classifier: 角色判别器；为 None 时事件角色为空.
            role_frame: 供角色判别的干净帧（未叠加标注）；
                为 None 时退回使用 frame.

        Returns:
            触发事件字典列表，包含 event / state / role / overlap_ratio 等字段.
        """
        events: List[Dict[str, Any]] = []
        best_overlap_ratio = 0.0
        best_distance = float("inf")
        triggered_bboxes: List[Sequence[float]] = []

        for hand_bbox in pointing_bboxes:
            hit = False
            for _, polygon in screen_polygons:
                overlap_ratio = bbox_polygon_overlap_ratio(hand_bbox, polygon)
                distance = bbox_to_polygon_distance(hand_bbox, polygon)
                best_overlap_ratio = max(best_overlap_ratio, overlap_ratio)
                best_distance = min(best_distance, distance)
                if (
                    overlap_ratio > self.screen_overlap_threshold
                    or distance < self.max_distance
                ):
                    hit = True
            if hit:
                triggered_bboxes.append(hand_bbox)

        if not triggered_bboxes:
            self._idle_frames += 1
            if self._idle_frames >= self.debounce_frames:
                self._active = False
            return events

        self._idle_frames = 0
        if self._active:
            return events

        # 上升沿：持续指向只记录一次，冷却仅作兜底
        self._active = True
        draw_trigger(frame, triggered_bboxes, "FINGER_SCREEN!")
        if not self._cooldown_ok(frame_count):
            return events

        self._mark_event(frame_count)
        role: Optional[str] = None
        if role_classifier is not None:
            role = role_classifier.classify(
                frame if role_frame is None else role_frame,
                triggered_bboxes,
            )
        events.append(
            self._make_event(
                "FINGER_SCREEN",
                "手指指向屏幕",
                frame_count,
                {
                    "overlap_ratio": round(float(best_overlap_ratio), 3),
                    "min_dist": round(float(best_distance), 1),
                },
                role=role,
            )
        )
        return events
