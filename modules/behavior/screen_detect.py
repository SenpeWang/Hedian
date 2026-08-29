"""手指指向屏幕检测模块.

基于 YOLO 目标检测输出的 pointing_hand 框与预标注屏幕/机箱多边形区域的重叠比及最短距离判定。
"""
import logging
from typing import Any, Dict, List

import numpy as np

from modules.behavior.base_detector import BaseDetector
from modules.behavior.behavior_utils import (
    bbox_polygon_overlap_ratio,
    bbox_to_polygon_distance,
)
from modules.behavior.behavior_vis import (
    SCREEN_POLYGONS,
    draw_detection_boxes,
    draw_roi_overlay,
    draw_trigger,
    screen_polygons_as_tuples,
)

logger = logging.getLogger("module.behavior.screen_detect")

# 类别索引（与 behavior_yolo.pt 训练一致）
CLASS_POINTING_HAND: int = 4

# 机箱多边形轮廓数组
_SCREEN_CONTOURS = [np.array(polygon, dtype=np.int32) for polygon in SCREEN_POLYGONS]


class FingerScreenDetector(BaseDetector):
    """手指指向屏幕行为检测器.

    使用 pointing_hand 框与机箱/屏幕 ROI 多边形的空间重叠度与几何距离进行判定。
    """

    def __init__(
        self,
        detect_conf: float = 0.25,
        screen_overlap_threshold: float = 0.2,
        max_distance: int = 20,
        cooldown_sec: float = 1.5,
        fps: float = 30.0,
    ) -> None:
        """初始化手指指向屏幕检测器.

        Args:
            detect_conf: 检测置信度阈值，默认 0.25.
            screen_overlap_threshold: 手指与屏幕区域面积重叠比阈值，默认 0.2.
            max_distance: 手指与屏幕多边形边缘的最大允许像素距离，默认 20.
            cooldown_sec: 动作触发后的冷却时间（秒），默认 1.5.
            fps: 视频采样帧率，默认 30.0.
        """
        super().__init__(cooldown_sec, fps)
        self.detect_conf: float = detect_conf
        self.screen_overlap_threshold: float = screen_overlap_threshold
        self.max_distance: int = max_distance

    @staticmethod
    def draw_roi(frame: np.ndarray) -> None:
        """在视频帧上绘制屏幕/机箱兴趣区域 (ROI) 多边形覆盖层.

        Args:
            frame: 原始视频帧 (BGR 格式).
        """
        draw_roi_overlay(frame, screen_polygons_as_tuples())

    def detect(
        self,
        frame: np.ndarray,
        results: Any,
        frame_count: int,
        fps: float,
    ) -> List[Dict[str, Any]]:
        """检测当前帧中是否存在手指向屏幕的行为.

        Args:
            frame: 当前视频帧（原地绘制标注与 ROI 框）.
            results: YOLO 模型推理结果对象.
            frame_count: 当前视频帧号.
            fps: 视频帧率.

        Returns:
            触发事件字典列表，包含 event, state, frame_id, overlap_ratio 等字段.
        """
        events: List[Dict[str, Any]] = []
        self.draw_roi(frame)

        if results is None:
            return events
        boxes = getattr(results, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return events

        boxes_xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        hands: List[Any] = []
        for index in range(len(classes)):
            if confidences[index] < self.detect_conf:
                continue
            if classes[index] == CLASS_POINTING_HAND:
                hands.append(boxes_xyxy[index])

        draw_detection_boxes(frame, hands=hands)

        # 与每个机箱多边形比较：重叠比 > 阈值 或 最短距离 <= max_distance 判定为命中
        triggered = False
        best_overlap_ratio = 0.0
        best_distance = 1e9
        for hand_box in hands:
            for screen_polygon in _SCREEN_CONTOURS:
                overlap_ratio = bbox_polygon_overlap_ratio(hand_box, screen_polygon)
                distance = bbox_to_polygon_distance(hand_box, screen_polygon)
                best_overlap_ratio = max(best_overlap_ratio, overlap_ratio)
                best_distance = min(best_distance, distance)
                if (
                    overlap_ratio >= self.screen_overlap_threshold
                    or distance <= self.max_distance
                ):
                    triggered = True

        if triggered:
            draw_trigger(frame, hands, "FINGER_SCREEN!")
            if self._cooldown_ok(frame_count):
                self._mark_event(frame_count)
                events.append(
                    self._make_event(
                        "FINGER_SCREEN",
                        "手指指向屏幕",
                        frame_count,
                        {
                            "overlap_ratio": round(float(best_overlap_ratio), 3),
                            "min_dist": round(float(best_distance), 1),
                        },
                    )
                )
        return events
