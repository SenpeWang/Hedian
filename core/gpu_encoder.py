"""
GPU / PyTorch Tensor 硬件加速图像编码器模块
"""
import torch
import torchvision
import cv2
import base64
import numpy as np
import logging

logger = logging.getLogger("core.gpu_encoder")


class GPUJPEGEncoder:
    """GPU / PyTorch 硬件加速 JPEG 编码器"""

    def __init__(self, target_width: int = 960, target_height: int = 540, quality: int = 40):
        self.target_width = target_width
        self.target_height = target_height
        self.quality = quality
        logger.info(f"GPUJPEGEncoder 初始化完成: target=({target_width}x{target_height}), quality={quality}")

    def encode_b64(self, frame_bgr: np.ndarray) -> str:
        """输入 BGR numpy 图像，通过 PyTorch/torchvision Tensor 加速转为 Base64 JPEG 字符串"""
        try:
            h, w, c = frame_bgr.shape
            if w > self.target_width or h > self.target_height:
                frame_bgr = cv2.resize(frame_bgr, (self.target_width, self.target_height))

            # BGR (H, W, C) -> RGB (C, H, W) Tensor
            rgb_tensor = torch.from_numpy(frame_bgr[:, :, ::-1].copy()).permute(2, 0, 1)
            jpeg_tensor = torchvision.io.encode_jpeg(rgb_tensor, quality=self.quality)
            return base64.b64encode(jpeg_tensor.numpy().tobytes()).decode("ascii")
        except Exception as e:
            logger.error(f"GPUJPEGEncoder 编码失败: {e}")
            # 降级备用
            _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            return base64.b64encode(buf.tobytes()).decode("ascii")
