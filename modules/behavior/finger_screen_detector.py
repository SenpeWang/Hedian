"""
手指屏幕检测模块

检测手指指向屏幕的行为。
"""
import os
import logging
import math
from typing import List, Dict, Optional

import numpy as np
import cv2

logger = logging.getLogger("module.behavior.finger_screen")

# 类别常量
CLASS_HAND = 0
CLASS_FILE = 1
CLASS_SCREEN = 2


class PoseEMAFilter:
    """姿态 EMA 平滑滤波器"""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.history = {}

    def update(self, track_id: int, keypoints: np.ndarray) -> np.ndarray:
        """
        更新关键点

        Args:
            track_id: 跟踪 ID
            keypoints: 关键点

        Returns:
            平滑后的关键点
        """
        if track_id == -1 or track_id not in self.history:
            self.history[track_id] = keypoints.copy()
            return keypoints

        smoothed_keypoints = np.zeros_like(keypoints)
        for i in range(len(keypoints)):
            curr_x, curr_y, curr_conf = keypoints[i]
            prev_x, prev_y, prev_conf = self.history[track_id][i]

            if curr_conf < 0.3:
                smoothed_keypoints[i] = [prev_x, prev_y, prev_conf]
            else:
                smooth_x = self.alpha * curr_x + (1 - self.alpha) * prev_x
                smooth_y = self.alpha * curr_y + (1 - self.alpha) * prev_y
                smoothed_keypoints[i] = [smooth_x, smooth_y, curr_conf]

        self.history[track_id] = smoothed_keypoints.copy()
        return smoothed_keypoints


def get_area(k1: np.ndarray, k2: np.ndarray, k3: np.ndarray) -> float:
    """计算三角形面积"""
    if k1[2] < 0.5 or k2[2] < 0.5 or k3[2] < 0.5:
        return 0
    return 0.5 * abs(
        k1[0] * (k2[1] - k3[1])
        + k2[0] * (k3[1] - k1[1])
        + k3[0] * (k1[1] - k2[1])
    )


class FingerScreenDetector:
    """
    手指屏幕检测器

    检测手指指向屏幕的行为。
    """

    def __init__(
        self,
        pose_model_path: str,
        finger_model_path: str,
    ):
        """
        初始化手指屏幕检测器

        Args:
            pose_model_path: 姿态模型路径
            finger_model_path: 手指检测模型路径
        """
        from ultralytics import YOLO

        # 加载模型
        if not os.path.exists(pose_model_path):
            raise FileNotFoundError(f"姿态模型不存在: {pose_model_path}")
        if not os.path.exists(finger_model_path):
            raise FileNotFoundError(f"手指检测模型不存在: {finger_model_path}")

        self._pose_model = YOLO(pose_model_path)
        self._finger_model = YOLO(finger_model_path)

        # 初始化滤波器
        self._pose_filter = PoseEMAFilter(alpha=0.5)

        # 事件冷却
        self._event_cooldown = {}
        self._cooldown_frames = 0

        logger.info(f"加载姿态模型: {os.path.basename(pose_model_path)}")
        logger.info(f"加载手指检测模型: {os.path.basename(finger_model_path)}")

    def detect(self, frame: np.ndarray, frame_count: int, fps: float) -> List[Dict]:
        """
        检测手指屏幕行为

        Args:
            frame: BGR 图像
            frame_count: 帧号
            fps: 帧率

        Returns:
            事件列表
        """
        if self._cooldown_frames == 0:
            self._cooldown_frames = int(fps * 1.5)

        events = []

        # 手指检测
        result_detect = self._finger_model.track(
            frame,
            persist=True,
            conf=0.3,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        # 姿态检测
        results_pose = self._pose_model.track(
            frame,
            persist=True,
            conf=0.5,
            tracker="bytetrack.yaml",
            iou=0.5,
            classes=[0],
            verbose=False,
        )

        # 解析检测结果
        hands = []
        screens = []

        if result_detect[0].boxes is not None:
            for box in result_detect[0].boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                track_id = int(box.id[0]) if box.id is not None else -1

                if cls_id == CLASS_HAND:
                    hands.append((cx, cy, x1, y1, x2, y2, track_id))
                elif cls_id == CLASS_SCREEN:
                    screens.append((cx, cy, x1, y1, x2, y2, track_id))

        # 解析姿态结果
        kpts = None
        pose_boxes = None
        pose_ids = None
        id_to_pose_box = {}

        if results_pose[0].keypoints is not None and len(results_pose[0].keypoints) > 0:
            kpts = results_pose[0].keypoints.data.cpu().numpy()
            pose_boxes = results_pose[0].boxes.xyxy.cpu().numpy()
            pose_ids = (
                results_pose[0].boxes.id.cpu().numpy()
                if results_pose[0].boxes.id is not None
                else [-1] * len(kpts)
            )

            # EMA 平滑
            smoothed_kpts = []
            for i, raw_kp in enumerate(kpts):
                p_id = int(pose_ids[i])
                smooth_kp = self._pose_filter.update(p_id, raw_kp)
                smoothed_kpts.append(smooth_kp)
            kpts = np.array(smoothed_kpts)

            # 构建 id -> pose_box 映射
            for i, p in enumerate(kpts):
                pid = int(pose_ids[i])
                if pid != -1 and pose_boxes is not None:
                    pb = pose_boxes[i]
                    id_to_pose_box[pid] = [int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])]

        # 检测手指屏幕行为
        for (hx, hy, hx1, hy1, hx2, hy2, h_id) in hands:
            own_id = -1
            min_wrist_dist = float("inf")

            # 找最近的手腕
            if kpts is not None and pose_ids is not None:
                for i, p in enumerate(kpts):
                    p_id = int(pose_ids[i])
                    lw_x, lw_y, lw_conf = p[9]
                    rw_x, rw_y, rw_conf = p[10]

                    if lw_conf > 0.5:
                        dist = math.sqrt((hx - lw_x) ** 2 + (hy - lw_y) ** 2)
                        if dist < min_wrist_dist:
                            min_wrist_dist, own_id = dist, p_id

                    if rw_conf > 0.5:
                        dist = math.sqrt((hx - rw_x) ** 2 + (hy - rw_y) ** 2)
                        if dist < min_wrist_dist:
                            min_wrist_dist, own_id = dist, p_id

            # 找最近的屏幕
            target_screen_id = -1
            target_sx, target_sy = -1, -1
            min_screen_dist = float("inf")

            for (sx, sy, sx1, sy1, sx2, sy2, s_id) in screens:
                dist = math.sqrt((hx - sx) ** 2 + (hy - sy) ** 2)
                if dist < min_screen_dist:
                    min_screen_dist = dist
                    target_screen_id = s_id
                    target_sx, target_sy = sx, sy

            # 判断是否指向屏幕
            if (
                own_id != -1
                and target_screen_id != -1
                and min_wrist_dist < 400
            ):
                # 冷却期检查
                key = ("手指屏幕", own_id)
                if frame_count - self._event_cooldown.get(key, -9999) > self._cooldown_frames:
                    person_box = id_to_pose_box.get(
                        own_id, [int(hx1), int(hy1), int(hx2), int(hy2)]
                    )

                    event = {
                        "event": "FINGER_SCREEN",
                        "state": "手指屏幕",
                        "person_id": own_id,
                        "screen_id": target_screen_id,
                        "person_box": person_box,
                        "frame_id": frame_count,
                    }
                    events.append(event)

                    self._event_cooldown[key] = frame_count

        return events
