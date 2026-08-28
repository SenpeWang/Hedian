"""头部检测模块.

使用 YOLOv8 ONNX 模型检测头部。
"""
import logging
import os
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger("module.gaze.head_detector")


class HeadBox:
    """头部边界框.

    Attributes:
        score: 检测置信度。
        x1: 左上角 x 坐标。
        y1: 左上角 y 坐标。
        x2: 右下角 x 坐标。
        y2: 右下角 y 坐标。
    """

    def __init__(self, score: float, x1: int, y1: int, x2: int, y2: int) -> None:
        """初始化头部边界框.

        Args:
            score: 检测置信度。
            x1: 左上角 x 坐标。
            y1: 左上角 y 坐标。
            x2: 右下角 x 坐标。
            y2: 右下角 y 坐标。
        """
        self.score = score
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def cx(self) -> int:
        """中心点 x 坐标.

        Returns:
            中心点 x 坐标（整数）。
        """
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        """中心点 y 坐标.

        Returns:
            中心点 y 坐标（整数）。
        """
        return (self.y1 + self.y2) // 2


class HeadDetector:
    """头部检测器.

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
    ) -> None:
        """初始化头部检测器.

        Args:
            model_path: 模型路径。
            conf_threshold: 置信度阈值。
            head_min_size: 最小头部大小。
            head_max_size: 最大头部大小。
            nms_iou_threshold: NMS IOU 阈值。
            providers: ONNX Runtime 提供者。
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLOv8 模型不存在: {model_path}")

        import onnxruntime
        # 官方 API: 从 pip nvidia 包预加载 CUDA/cuDNN 库，供下方 CUDA EP 使用
        onnxruntime.preload_dlls()
        if providers is None:
            providers = ["CUDAExecutionProvider"]

        # 仅使用 GPU
        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 3
        onnxruntime.set_default_logger_severity(3)

        self._session = onnxruntime.InferenceSession(
            model_path, sess_options=session_options, providers=providers
        )
        # 硬性要求: 仅允许 GPU 推理; CUDA EP 未生效说明已静默回退 CPU, 立即失败
        if "CUDAExecutionProvider" not in self._session.get_providers():
            raise RuntimeError(
                f"YOLOv8 头部检测模型 CUDAExecutionProvider 未生效(实际: {self._session.get_providers()}), "
                "按约定禁止 CPU 回退, 拒绝启动"
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
        """检测头部.

        Args:
            image: BGR 图像。

        Returns:
            头部边界框列表。
        """
        image_h, image_w = image.shape[:2]

        # Letterbox 缩放：保持长宽比，灰色填充
        scale = min(self._input_size / image_h, self._input_size / image_w)
        new_width = int(image_w * scale)
        new_height = int(image_h * scale)
        resized = cv2.resize(image, (new_width, new_height),
                             interpolation=cv2.INTER_LINEAR)
        padded = np.full((self._input_size, self._input_size, 3), 114, dtype=np.uint8)
        pad_x = (self._input_size - new_width) // 2
        pad_y = (self._input_size - new_height) // 2
        padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized

        # BGR 转 RGB、归一化并转 CHW
        input_tensor = padded[:, :, ::-1].astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})

        # 输出 [1, 5, 8400] 转置为 [8400, 5] = [center_x, center_y, width, height, score]
        predictions = outputs[0][0].T

        scores = predictions[:, 4]
        valid_mask = scores > self._conf_threshold
        predictions = predictions[valid_mask]

        if len(predictions) == 0:
            return []

        # 由中心点与宽高换算边界框（padding 图坐标）
        center_x, center_y = predictions[:, 0], predictions[:, 1]
        head_width, head_height = predictions[:, 2], predictions[:, 3]
        x1 = center_x - head_width / 2
        y1 = center_y - head_height / 2
        x2 = center_x + head_width / 2
        y2 = center_y + head_height / 2

        # NMS
        nms_boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        nms_scores = predictions[:, 4].astype(np.float32)
        indices = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(),
            nms_scores.tolist(),
            self._conf_threshold,
            self._nms_iou_threshold,
        )

        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()

        # 去除 padding 偏移并缩放到原图坐标
        heads = []
        for index in indices:
            box_x1 = int((x1[index] - pad_x) / scale)
            box_y1 = int((y1[index] - pad_y) / scale)
            box_x2 = int((x2[index] - pad_x) / scale)
            box_y2 = int((y2[index] - pad_y) / scale)

            box_x1 = max(0, box_x1)
            box_y1 = max(0, box_y1)
            box_x2 = min(image_w, box_x2)
            box_y2 = min(image_h, box_y2)

            box_width = box_x2 - box_x1
            box_height = box_y2 - box_y1

            if box_width < self._head_min_size or box_height < self._head_min_size:
                continue
            if box_width > self._head_max_size or box_height > self._head_max_size:
                continue

            heads.append(
                HeadBox(
                    score=float(nms_scores[index]),
                    x1=box_x1,
                    y1=box_y1,
                    x2=box_x2,
                    y2=box_y2,
                )
            )

        return heads
