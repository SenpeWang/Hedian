"""
头部检测模块

使用 YOLOv8 ONNX 模型检测头部。
"""
import os
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import cv2

logger = logging.getLogger("module.gaze.head_detector")


class HeadBox:
    """头部边界框"""

    def __init__(self, score: float, x1: int, y1: int, x2: int, y2: int):
        self.score = score
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def cx(self) -> int:
        """中心点 x 坐标"""
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        """中心点 y 坐标"""
        return (self.y1 + self.y2) // 2


class HeadDetector:
    """
    头部检测器

    使用 YOLOv8 ONNX 模型检测头部。
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.55,
        head_min_size: int = 20,
        head_max_size: int = 300,
        nms_iou_threshold: float = 0.45,
        providers: Optional[List[str]] = None,
    ):
        """
        初始化头部检测器

        Args:
            model_path: 模型路径
            conf_threshold: 置信度阈值
            head_min_size: 最小头部大小
            head_max_size: 最大头部大小
            nms_iou_threshold: NMS IOU 阈值
            providers: ONNX Runtime 提供者
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLOv8 模型不存在: {model_path}")

        import onnxruntime

        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_opts = onnxruntime.SessionOptions()
        sess_opts.log_severity_level = 3
        onnxruntime.set_default_logger_severity(3)

        self._session = onnxruntime.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._conf_threshold = conf_threshold
        self._head_min_size = head_min_size
        self._head_max_size = head_max_size
        self._nms_iou_threshold = nms_iou_threshold
        self._input_size = 640

        logger.info(f"加载头部检测模型: {os.path.basename(model_path)}")

    def detect(self, image: np.ndarray) -> List[HeadBox]:
        """
        检测头部

        Args:
            image: BGR 图像

        Returns:
            头部边界框列表
        """
        h, w = image.shape[:2]

        # Letterbox resize (maintain aspect ratio, pad with gray)
        scale = min(self._input_size / h, self._input_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self._input_size, self._input_size, 3), 114, dtype=np.uint8)
        pad_x = (self._input_size - new_w) // 2
        pad_y = (self._input_size - new_h) // 2
        padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        # BGR to RGB, normalize, CHW
        input_tensor = padded[:, :, ::-1].astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})

        # Output: [1, 5, 8400] → transpose to [8400, 5] = [cx, cy, w, h, score]
        preds = outputs[0][0].T

        scores = preds[:, 4]
        mask = scores > self._conf_threshold
        preds = preds[mask]

        if len(preds) == 0:
            return []

        # Convert cx,cy,w,h to x1,y1,x2,y2 (in padded image scale)
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # NMS
        boxes_for_nms = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        scores_for_nms = preds[:, 4].astype(np.float32)
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(),
            scores_for_nms.tolist(),
            self._conf_threshold,
            self._nms_iou_threshold,
        )

        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()

        # Remove padding offset and scale to original image coordinates
        heads = []
        for idx in indices:
            bx1 = int((x1[idx] - pad_x) / scale)
            by1 = int((y1[idx] - pad_y) / scale)
            bx2 = int((x2[idx] - pad_x) / scale)
            by2 = int((y2[idx] - pad_y) / scale)

            bx1 = max(0, bx1)
            by1 = max(0, by1)
            bx2 = min(w, bx2)
            by2 = min(h, by2)

            box_w = bx2 - bx1
            box_h = by2 - by1

            if box_w < self._head_min_size or box_h < self._head_min_size:
                continue
            if box_w > self._head_max_size or box_h > self._head_max_size:
                continue

            heads.append(HeadBox(
                score=float(scores_for_nms[idx]),
                x1=bx1,
                y1=by1,
                x2=bx2,
                y2=by2,
            ))

        return heads
