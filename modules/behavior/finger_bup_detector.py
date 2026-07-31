"""
手指屏幕检测模块

检测手指指向屏幕的行为。
推理流：在帧上绘制检测框标注（手/屏幕/人体）。
事件流：关键事件（手指屏幕）存入 _events。
"""
import os
import logging
import math
from typing import List, Dict

import numpy as np
import cv2

logger = logging.getLogger("module.behavior.finger_screen")

# 类别常量
CLASS_HAND = 0
CLASS_SCREEN = 2

# 绘制颜色 (BGR)
COLOR_HAND = (0, 200, 0)       # 绿色 - 手
COLOR_SCREEN = (0, 200, 0)     # 绿色 - 屏幕
COLOR_PERSON = (0, 200, 255)   # 黄色 - 人体
COLOR_EVENT = (0, 0, 255)      # 红色 - 事件触发


class PoseEMAFilter:
    """姿态 EMA 平滑滤波器"""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.history = {}

    def update(self, track_id: int, keypoints: np.ndarray) -> np.ndarray:
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


class FingerBupDetector:
    """
    手指屏幕检测器

    推理流：每帧绘制检测框（手/屏幕/人体）。
    事件流：关键事件（手指屏幕）触发时生成。
    """

    def __init__(
        self,
        pose_model_path: str,
        finger_model_path: str,
        detect_conf: float = 0.3,
        pose_conf: float = 0.5,
        hand_to_screen_dist: float = 400,
        cooldown_sec: float = 1.5,
    ):
        from ultralytics import YOLO

        if not os.path.exists(pose_model_path):
            raise FileNotFoundError(f"姿态模型不存在: {pose_model_path}")
        if not os.path.exists(finger_model_path):
            raise FileNotFoundError(f"手指检测模型不存在: {finger_model_path}")

        self._pose_model = YOLO(pose_model_path)
        self._finger_model = YOLO(finger_model_path)

        self._detect_conf = detect_conf
        self._pose_conf = pose_conf
        self._hand_to_screen_dist = hand_to_screen_dist
        self._cooldown_sec = cooldown_sec

        self._pose_filter = PoseEMAFilter(alpha=0.5)
        self._event_cooldown = {}
        self._cooldown_frames = 0

        logger.info(f"加载姿态模型: {os.path.basename(pose_model_path)}")
        logger.info(f"加载手指检测模型: {os.path.basename(finger_model_path)}")

    def detect(self, frame: np.ndarray, frame_count: int, fps: float) -> List[Dict]:
        """
        检测手指屏幕行为，同时在帧上绘制推理流标注。

        Args:
            frame: BGR 图像（会被原地绘制标注）
            frame_count: 帧号
            fps: 帧率

        Returns:
            事件列表
        """
        if self._cooldown_frames == 0:
            self._cooldown_frames = int(fps * self._cooldown_sec)

        events = []

        # 手指检测
        result_detect = self._finger_model.track(
            frame, persist=True, conf=self._detect_conf,
            tracker="bytetrack.yaml", half=True, verbose=False,
        )

        # 姿态检测
        results_pose = self._pose_model.track(
            frame, persist=True, conf=self._pose_conf,
            tracker="bytetrack.yaml", iou=0.5, classes=[0], verbose=False,
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

            smoothed_kpts = []
            for i, raw_kp in enumerate(kpts):
                p_id = int(pose_ids[i])
                smooth_kp = self._pose_filter.update(p_id, raw_kp)
                smoothed_kpts.append(smooth_kp)
            kpts = np.array(smoothed_kpts)

            for i, p in enumerate(kpts):
                pid = int(pose_ids[i])
                if pid != -1 and pose_boxes is not None:
                    pb = pose_boxes[i]
                    id_to_pose_box[pid] = [int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])]

        # 检测手指屏幕行为
        event_person_ids = set()
        for (hx, hy, hx1, hy1, hx2, hy2, h_id) in hands:
            own_id = -1
            min_wrist_dist = float("inf")

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

            target_screen_id = -1
            min_screen_dist = float("inf")

            for (sx, sy, sx1, sy1, sx2, sy2, s_id) in screens:
                dist = math.sqrt((hx - sx) ** 2 + (hy - sy) ** 2)
                if dist < min_screen_dist:
                    min_screen_dist = dist
                    target_screen_id = s_id

            if (
                own_id != -1
                and target_screen_id != -1
                and min_wrist_dist < self._hand_to_screen_dist
            ):
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
                    event_person_ids.add(own_id)
                    self._event_cooldown[key] = frame_count

        # === 推理流：绘制检测框标注 ===
        self._draw_frame(frame, hands, screens, id_to_pose_box, event_person_ids)

        return events

    def _draw_frame(self, frame, hands, screens, id_to_pose_box, event_person_ids):
        """在帧上绘制推理流标注（检测框 + 标签）"""
        # 绘制屏幕框（青色）
        for (sx, sy, sx1, sy1, sx2, sy2, s_id) in screens:
            cv2.rectangle(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), COLOR_SCREEN, 2)
            cv2.putText(frame, f"Screen#{s_id}", (int(sx1), int(sy1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SCREEN, 1)

        # 绘制人体框（黄色，事件触发时红色加粗）
        for pid, box in id_to_pose_box.items():
            x1, y1, x2, y2 = box
            if pid in event_person_ids:
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_EVENT, 3)
                cv2.putText(frame, "FINGER_SCREEN!", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_EVENT, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 1)
                cv2.putText(frame, f"Person#{pid}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_PERSON, 1)

        # 绘制手部框（绿色）
        for (hx, hy, hx1, hy1, hx2, hy2, h_id) in hands:
            cv2.rectangle(frame, (int(hx1), int(hy1)), (int(hx2), int(hy2)), COLOR_HAND, 2)
            cv2.putText(frame, f"Hand#{h_id}", (int(hx1), int(hy2) + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HAND, 1)
