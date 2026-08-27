"""手指指向文件检测模块.

基于 YOLO 目标检测输出的 pointing_hand 框与 file 框的 IoU 几何交叠判定手指向文件行为。
"""
import logging
from typing import Any, Dict, List

import numpy as np

from modules.behavior.base_detector import BaseDetector
from modules.behavior.behavior_utils import bbox_iou
from modules.behavior.behavior_vis import (
    draw_detection_boxes,
    draw_trigger,
)

logger = logging.getLogger("module.behavior.file_detector")

# 类别索引（与 behavior_yolo.pt 训练一致）
CLS_FILE: int = 1
CLS_POINTING_HAND: int = 4


class FingerFileDetector(BaseDetector):
    """手指指向文件行为检测器.

    使用 pointing_hand(4) 与 file(1) 边界框的 IoU 重叠度进行判定，结合冷却机制避免高频抖动。
    """

    def __init__(
        self,
        detect_conf: float = 0.25,
        file_iou_threshold: float = 0.2,
        cooldown_sec: float = 1.5,
        fps: float = 30.0,
    ):
        """初始化手指指向文件检测器.

        Args:
            detect_conf (float): 检测置信度阈值，默认 0.25.
            file_iou_threshold (float): 手指框与文件框的判定 IoU 阈值，默认 0.2.
            cooldown_sec (float): 动作触发后的冷却时间（秒），默认 1.5.
            fps (float): 视频采样帧率，默认 30.0.
        """
        super().__init__(cooldown_sec, fps)
        self.detect_conf: float = detect_conf
        self.file_iou_threshold: float = file_iou_threshold

    def detect(
        self,
        frame: np.ndarray,
        results: Any,
        frame_count: int,
        fps: float,
    ) -> List[Dict[str, Any]]:
        """检测当前帧中是否存在手指向文件的行为.

        Args:
            frame (np.ndarray): 当前视频帧（原地绘制检测框）.
            results (Any): YOLO 模型推理结果对象.
            frame_count (int): 当前视频帧号.
            fps (float): 视频帧率.

        Returns:
            List[Dict[str, Any]]: 触发事件字典列表，包含 event, state, frame_id, iou 等字段.
        """
        events: List[Dict[str, Any]] = []
        if results is None:
            return events
        det = getattr(results, "boxes", None)
        if det is None or len(det) == 0:
            return events

        boxes_xyxy = det.xyxy.cpu().numpy()
        classes = det.cls.cpu().numpy().astype(int)
        confidences = det.conf.cpu().numpy()

        hands, files = [], []
        for index in range(len(classes)):
            if confidences[index] < self.detect_conf:
                continue
            box = boxes_xyxy[index]
            if classes[index] == CLS_POINTING_HAND:
                hands.append(box)
            elif classes[index] == CLS_FILE:
                files.append(box)

        draw_detection_boxes(frame, hands=hands, files=files)

        # 指向文件判定：pointing_hand 与 file 框 IoU 超过阈值
        triggered = False
        best_iou = 0.0
        for hand_box in hands:
            for file_box in files:
                iou_val = bbox_iou(hand_box, file_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                if iou_val >= self.file_iou_threshold:
                    triggered = True

        if triggered:
            draw_trigger(frame, hands, "FINGER_FILE!")

            if self._cooldown_ok(frame_count):
                self._mark_event(frame_count)
                events.append(
                    self._make_event(
                        "FINGER_FILE",
                        "手指指向文件",
                        frame_count,
                        {"iou": round(float(best_iou), 3)},
                    )
                )
        return events
