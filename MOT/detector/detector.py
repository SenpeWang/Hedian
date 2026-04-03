"""
目标检测器 — YOLO 目标检测 + YOLOPose 骨架估计
"""
import os
import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO
from typing import List, Dict, Optional

_HEDIAN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DATA_ROOT = os.path.abspath(os.path.join(_HEDIAN, "..", "Hedian_data"))


@dataclass
class DetectionConfig:
    """检测器配置"""
    model_path: str = os.path.join(_DATA_ROOT, "yolo_model", "best.pt")
    pose_model_path: str = os.path.join(_DATA_ROOT, "yolo_model", "yolo26s-pose.pt")
    conf_threshold: float = 0.65
    pose_confidence: float = 0.6
    nms_threshold: float = 0.35
    img_size: int = 640


class ObjectDetector:
    """
    目标检测器

    - detect(frame)           → YOLO person 框
    - detect_with_pose(frame) → 同 detect，但同时跑 Pose（可选）
    - detect_pose(frame)      → 仅 Pose 检测，返回骨架列表
    - check_hand_raised(kps)  → 手腕Y < 肩膀Y → 举手
    """

    def __init__(self, model_path: str,
                 pose_model_path: str = None,
                 conf_threshold: float = 0.65,
                 pose_confidence: float = 0.3,
                 nms_threshold: float = 0.35,
                 img_size: int = 640):
        self.conf_threshold = conf_threshold
        self.pose_confidence = pose_confidence
        self.nms_threshold = nms_threshold
        self.img_size = img_size

        self.model = YOLO(model_path)
        print(f"[Detector] 加载检测模型: {os.path.basename(model_path)}")

        self.pose_model = None
        if pose_model_path and os.path.exists(pose_model_path):
            self.pose_model = YOLO(pose_model_path)
            print(f"[Detector] 加载姿态模型: {os.path.basename(pose_model_path)}")

    # ─────────── 基础检测 ───────────
    def detect(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.nms_threshold,
            imgsz=self.img_size,
            verbose=False,
            save=False,
        )
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                if class_id == 0:  # person
                    detections.append({
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": confidence,
                        "class": "person",
                        "class_id": class_id,
                    })
        return detections

    def detect_with_pose(self, frame: np.ndarray) -> List[Dict]:
        """目标检测（同 detect），不主动做 Pose"""
        return self.detect(frame)

    # ─────────── 姿态估计 ───────────
    def detect_pose(self, frame: np.ndarray) -> List[Dict]:
        """
        YOLOPose 骨架估计，返回列表 [{center, keypoints, box}]
        keypoints 索引：5/6=肩膀，9/10=手腕
        """
        if self.pose_model is None:
            return []
        results = self.pose_model(
            frame,
            conf=self.pose_confidence,
            imgsz=self.img_size,
            verbose=False,
            save=False,
        )
        poses = []
        for result in results:
            if result.keypoints is None:
                continue
            kps_data = result.keypoints.data.cpu().numpy()   # (N, 17, 3)
            boxes = result.boxes
            for i in range(len(kps_data)):
                kps = kps_data[i]                           # (17, 3)
                bx = boxes[i].xyxy[0].cpu().numpy() if boxes is not None else [0,0,0,0]
                cx = (float(bx[0]) + float(bx[2])) / 2
                cy = (float(bx[1]) + float(bx[3])) / 2
                poses.append({
                    "center": np.array([cx, cy]),
                    "keypoints": kps,
                    "box": [float(b) for b in bx],
                })
        return poses

    # ─────────── 举手判断 ───────────
    @staticmethod
    def check_hand_raised(keypoints: np.ndarray, conf_thr: float = 0.3) -> bool:
        """
        举手检测：适度宽松 + 投票机制（在 main.py 中）防误判。
        keypoints: (17, 3) — x, y, conf
        关键点索引: 0鼻 5左肩 6右肩 7左肘 8右肘 9左腕 10右腕
        """
        def _conf_ok(*idxs):
            return all(keypoints[i][2] >= conf_thr for i in idxs)

        WRIST_MARGIN = 15  # 手腕需高于肩膀（像素）

        # 条件1: 手腕高于同侧肩膀 15px 以上（经典举手）
        for sh, wr in [(5, 9), (6, 10)]:
            if _conf_ok(sh, wr):
                if keypoints[wr][1] < keypoints[sh][1] - WRIST_MARGIN:
                    return True

        # 条件2: 手肘高于同侧肩膀 10px（手臂明显抬起）
        for sh, el in [(5, 7), (6, 8)]:
            if _conf_ok(sh, el):
                if keypoints[el][1] < keypoints[sh][1] - 10:
                    return True

        # 条件3: 手腕高于鼻子（高举过头）
        if _conf_ok(0):
            for wr in [9, 10]:
                if _conf_ok(wr):
                    if keypoints[wr][1] < keypoints[0][1]:
                        return True

        return False

