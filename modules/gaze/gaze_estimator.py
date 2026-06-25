"""
注视推断模块

使用 Gazelle ONNX 模型推断注视方向。
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2

from modules.gaze.head_detector import HeadBox

logger = logging.getLogger("module.gaze.estimator")


class GazeEstimator:
    """
    注视推断器

    使用 Gazelle ONNX 模型推断注视方向。
    """

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
    ):
        """
        初始化注视推断器

        Args:
            model_path: 模型路径
            providers: ONNX Runtime 提供者
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Gazelle 模型不存在: {model_path}")

        import onnxruntime

        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_opts = onnxruntime.SessionOptions()
        sess_opts.log_severity_level = 3

        self._session = onnxruntime.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]

        self._heatmap_idx = (
            self._output_names.index("heatmap")
            if "heatmap" in self._output_names
            else 0
        )
        self._inout_idx = (
            self._output_names.index("inout")
            if "inout" in self._output_names
            else None
        )

        logger.info(f"加载注视推断模型: {os.path.basename(model_path)}")

    def predict(
        self,
        image: np.ndarray,
        head_boxes: List[HeadBox],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[HeadBox]]:
        """
        推断注视方向

        Args:
            image: BGR 图像
            head_boxes: 头部边界框列表

        Returns:
            (heatmaps[N,H,W], inout_scores[N], valid_boxes)
        """
        if not head_boxes:
            return None, None, []

        h, w = image.shape[:2]
        inv_w = 1.0 / float(w)
        inv_h = 1.0 / float(h)

        normalized_boxes = []
        valid_boxes = []

        for box in head_boxes:
            x1 = np.clip(box.x1 * inv_w, 0.0, 1.0)
            y1 = np.clip(box.y1 * inv_h, 0.0, 1.0)
            x2 = np.clip(box.x2 * inv_w, 0.0, 1.0)
            y2 = np.clip(box.y2 * inv_h, 0.0, 1.0)

            if x2 <= x1 or y2 <= y1:
                continue

            normalized_boxes.append([x1, y1, x2, y2])
            valid_boxes.append(box)

        if not normalized_boxes:
            return None, None, []

        resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
        image_tensor = resized.transpose(2, 0, 1).astype(np.float32)
        image_input = np.expand_dims(image_tensor, axis=0)
        bbox_input = np.array([normalized_boxes], dtype=np.float32)

        feed = {
            self._input_names[0]: image_input,
            self._input_names[1]: bbox_input,
        }

        outputs = self._session.run(self._output_names, feed)

        heatmaps = outputs[self._heatmap_idx]
        inout_scores = (
            outputs[self._inout_idx] if self._inout_idx is not None else None
        )

        return heatmaps, inout_scores, valid_boxes
