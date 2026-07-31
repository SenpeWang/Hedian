"""
手指文件检测模块

检测手指指向文件的行为。
使用自训练 YOLO 模型检测 person / file / pointing_hand 三类目标，
通过 pointing_hand 与 file 的 IOU 判断是否指向文件。
推理流：在帧上绘制检测框标注（人/文件/手指）。
事件流：关键事件（指向文件）存入 _events。
"""
import os
import math
import logging
from typing import List, Dict

import numpy as np
import cv2

logger = logging.getLogger("module.behavior.finger_file")

# 类别常量
CLASS_PERSON = 0
CLASS_FILE = 1
CLASS_POINTING_HAND = 2

# 绘制颜色 (BGR)
COLOR_PERSON = (0, 200, 255)   # 黄色 - 人
COLOR_FILE = (0, 200, 0)       # 绿色 - 文件
COLOR_HAND = (0, 200, 0)       # 绿色 - 手指
COLOR_EVENT = (0, 0, 255)      # 红色 - 事件触发


class FingerFileDetector:
    """
    手指文件检测器

    推理流：每帧绘制检测框（人/文件/手指）。
    事件流：关键事件（指向文件）触发时生成。
    """

    def __init__(
        self,
        model_path: str,
        detect_conf: float = 0.25,
        track_iou: float = 0.5,
        file_iou_threshold: float = 0.2,
        cooldown_sec: float = 1.5,
    ):
        from ultralytics import YOLO

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self._model = YOLO(model_path)
        self._detect_conf = detect_conf
        self._track_iou = track_iou
        self._file_iou_threshold = file_iou_threshold
        self._cooldown_sec = cooldown_sec

        self._event_cooldown = {}
        self._cooldown_frames = 0

        logger.info(f"加载手指文件检测模型: {os.path.basename(model_path)}")

    @staticmethod
    def _center(bbox):
        """计算 bbox 中心点"""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    @staticmethod
    def _bbox_iou(a, b):
        """计算两个 bbox 的 IOU"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    @staticmethod
    def _contains_center(outer_bbox, inner_bbox):
        """判断 inner_bbox 中心是否在 outer_bbox 内"""
        cx, cy = FingerFileDetector._center(inner_bbox)
        x1, y1, x2, y2 = outer_bbox
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _nearest_person_id(self, hand_bbox, persons):
        """找到包含手指中心或距离最近的 person"""
        if not persons:
            return -1
        containing = [p for p in persons if self._contains_center(p["bbox"], hand_bbox)]
        candidates = containing or persons
        hx, hy = self._center(hand_bbox)
        best = min(
            candidates,
            key=lambda p: math.hypot(hx - self._center(p["bbox"])[0], hy - self._center(p["bbox"])[1]),
        )
        return best["id"]

    def detect(self, frame: np.ndarray, frame_count: int, fps: float) -> List[Dict]:
        """
        检测手指文件行为，同时在帧上绘制推理流标注。

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

        results = self._model.track(
            frame, persist=True, conf=self._detect_conf,
            iou=self._track_iou, tracker="bytetrack.yaml", half=True, verbose=False,
        )

        if not (results and results[0].boxes is not None):
            return events

        boxes = results[0].boxes
        detections = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].cpu().numpy()]
            track_id = int(boxes.id[i]) if boxes.id is not None else i
            detections.append({"cls": cls_id, "bbox": (x1, y1, x2, y2), "id": track_id})

        persons = [d for d in detections if d["cls"] == CLASS_PERSON]
        files = [d for d in detections if d["cls"] == CLASS_FILE]
        pointing_hands = [d for d in detections if d["cls"] == CLASS_POINTING_HAND]

        # 记录事件触发的手指-文件对（用于绘制连线）
        event_hand_file_pairs = []
        event_person_ids = set()

        for hand in pointing_hands:
            best_iou = 0.0
            best_file = None
            for f in files:
                iou = self._bbox_iou(hand["bbox"], f["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_file = f

            if best_file is not None and best_iou > self._file_iou_threshold:
                owner_id = self._nearest_person_id(hand["bbox"], persons)

                key = ("指向文件", owner_id)
                if frame_count - self._event_cooldown.get(key, -9999) > self._cooldown_frames:
                    self._event_cooldown[key] = frame_count
                    events.append({
                        "event": "FINGER_FILE",
                        "state": "指向文件",
                        "person_id": owner_id,
                        "file_iou": round(best_iou, 3),
                        "frame_id": frame_count,
                    })
                    event_hand_file_pairs.append((hand, best_file))
                    event_person_ids.add(owner_id)

        # === 推理流：绘制检测框标注 ===
        self._draw_frame(frame, persons, files, pointing_hands,
                         event_hand_file_pairs, event_person_ids)

        return events

    def _draw_frame(self, frame, persons, files, pointing_hands,
                    event_hand_file_pairs, event_person_ids):
        """在帧上绘制推理流标注（检测框 + 标签 + 事件连线）"""
        # 绘制文件框（青色）
        for f in files:
            x1, y1, x2, y2 = f["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_FILE, 2)
            cv2.putText(frame, f"File#{f['id']}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_FILE, 1)

        # 绘制人体框（黄色，事件触发时红色加粗）
        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            if p["id"] in event_person_ids:
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_EVENT, 3)
                cv2.putText(frame, "FINGER_FILE!", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_EVENT, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 1)
                cv2.putText(frame, f"Person#{p['id']}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_PERSON, 1)

        # 绘制手指框（绿色）
        for hand in pointing_hands:
            x1, y1, x2, y2 = hand["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HAND, 2)
            cv2.putText(frame, f"Hand#{hand['id']}", (x1, y2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HAND, 1)

        # 绘制事件连线（红色虚线：手指中心 → 文件中心）
        for hand, f in event_hand_file_pairs:
            hc = self._center(hand["bbox"])
            fc = self._center(f["bbox"])
            cv2.line(frame, (int(hc[0]), int(hc[1])), (int(fc[0]), int(fc[1])),
                     COLOR_EVENT, 2, cv2.LINE_AA)
