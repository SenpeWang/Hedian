"""目标检测与人体姿态估计模块.

基于 YOLO 系列模型提供目标检测（Person）与骨架关键点估计（Pose Estimation）能力。
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("module.tracker.detector")


class ObjectDetector:
    """目标检测与人体姿态估计器.

    封装 YOLO 目标检测模型与 YOLOPose 骨架估计模型，支持多阈值检测与举手骨架特征判定。
    """

    def __init__(
        self,
        model_path: str,
        pose_model_path: Optional[str] = None,
        conf_threshold: float = 0.65,
        pose_confidence: float = 0.3,
        nms_threshold: float = 0.35,
        img_size: int = 640,
    ) -> None:
        """初始化检测器.

        Args:
            model_path: YOLO 目标检测模型权重路径.
            pose_model_path: YOLOPose 姿态估计模型权重路径，可选.
            conf_threshold: 目标检测置信度阈值，默认 0.65.
            pose_confidence: 姿态估计置信度阈值，默认 0.3.
            nms_threshold: 非极大值抑制 (NMS) IoU 阈值，默认 0.35.
            img_size: 推理输入图像尺寸，默认 640.

        Raises:
            FileNotFoundError: 当检测模型文件不存在时抛出.
        """
        from ultralytics import YOLO

        self.conf_threshold: float = conf_threshold
        self.pose_confidence: float = pose_confidence
        self.nms_threshold: float = nms_threshold
        self.img_size: int = img_size

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"检测模型文件不存在: {model_path}")

        self.model = YOLO(model_path)
        logger.info(f"加载目标检测模型: {os.path.basename(model_path)}")

        self.pose_model = None
        if pose_model_path and os.path.exists(pose_model_path):
            self.pose_model = YOLO(pose_model_path)
            logger.info(f"加载姿态估计模型: {os.path.basename(pose_model_path)}")

    def detect_two_thresholds(
        self, frame: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """执行双阈值目标检测（专供 ByteTrack 高低分匹配阶段使用）.

        Args:
            frame: 输入的原始视频帧 (BGR 格式).

        Returns:
            按置信度拆分的高/低检测结果元组:
                - high_detections: 高置信度检测结果列表 (conf >= conf_threshold)
                - low_detections: 低置信度检测结果列表 (0.10 <= conf < conf_threshold)
        """
        results = self.model(
            frame,
            conf=0.10,
            iou=self.nms_threshold,
            imgsz=self.img_size,
            half=True,
            verbose=False,
            save=False,
        )

        high_detections: List[Dict[str, Any]] = []
        low_detections: List[Dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                if class_id == 0:  # person
                    detection_item = {
                        "box": [float(x_min), float(y_min), float(x_max), float(y_max)],
                        "confidence": confidence,
                        "class": "person",
                        "class_id": class_id,
                    }
                    if confidence >= self.conf_threshold:
                        high_detections.append(detection_item)
                    else:
                        low_detections.append(detection_item)

        return high_detections, low_detections

    def detect_pose(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """使用 YOLOPose 估计输入帧中所有人体骨架关键点.

        Args:
            frame: 输入的原始视频帧 (BGR 格式).

        Returns:
            姿态估计结果列表，每个元素包含：
                - 'center' (np.ndarray): 目标人体中心点坐标 [cx, cy]
                - 'keypoints' (np.ndarray): 17个关键点坐标与置信度数组 (17, 3)
                - 'box' (List[float]): 人体边界框 [x_min, y_min, x_max, y_max]
        """
        if self.pose_model is None:
            return []

        results = self.pose_model(
            frame,
            conf=self.pose_confidence,
            imgsz=self.img_size,
            half=True,
            verbose=False,
            save=False,
        )

        poses: List[Dict[str, Any]] = []
        for result in results:
            if result.keypoints is None:
                continue

            keypoints_data = result.keypoints.data.cpu().numpy()
            boxes = result.boxes

            for index in range(len(keypoints_data)):
                keypoints_array = keypoints_data[index]
                box_coords = (
                    boxes[index].xyxy[0].cpu().numpy()
                    if boxes is not None
                    else [0, 0, 0, 0]
                )
                center_x = (float(box_coords[0]) + float(box_coords[2])) / 2.0
                center_y = (float(box_coords[1]) + float(box_coords[3])) / 2.0

                poses.append({
                    "center": np.array([center_x, center_y]),
                    "keypoints": keypoints_array,
                    "box": [float(coord) for coord in box_coords],
                })

        return poses
