"""InsightFace 人脸特征提取与底库距离计算.

迁移自 Nuclear_autoface_enhanced 的 `robust_face_distance` 口径：
余弦相似度转欧氏距离 `sqrt(2 - 2cos)`，再对最近 K 个模板取均值（topk_mean），
避免单模板误判。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger("module.identity.face")


def face_distance(
    query: np.ndarray,
    templates: np.ndarray,
    aggregation: str = "topk_mean",
    k: int = 3,
) -> float:
    """计算查询人脸与模板库的距离.

    Args:
        query: 查询特征向量 (D,)，未归一化也可.
        templates: 模板库 (N, D)；空则返回 inf.
        aggregation: 'min' 取最近模板；'topk_mean' 取最近 k 个的均值.
        k: topk 的 K 值.

    Returns:
        距离值；模板库为空返回 float('inf').
    """
    if templates is None:
        return float("inf")
    tpl = np.asarray(templates, dtype=np.float32)
    if tpl.size == 0:
        return float("inf")
    if tpl.ndim == 1:
        tpl = tpl.reshape(1, -1)
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    q = q / (np.linalg.norm(q) + 1e-8)
    tpl_norm = tpl / (np.linalg.norm(tpl, axis=1, keepdims=True) + 1e-8)
    cos_sims = np.clip(tpl_norm @ q, -1.0, 1.0)
    distances = np.sqrt(2.0 - 2.0 * cos_sims)
    if aggregation == "min":
        return float(np.min(distances))
    k_eff = max(1, min(int(k), len(distances)))
    if k_eff == len(distances):
        return float(np.mean(distances))
    top_k = np.partition(distances, k_eff - 1)[:k_eff]
    return float(np.mean(top_k))


class FaceRecognizer:
    """人脸检测 + 特征提取封装.

    只在 `extract()` 调用时跑推理，调用方负责节流（建议 0.5s 一次/人）。
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: int = 640,
        ctx_id: int = 0,
        min_box_size: int = 40,
        det_score: float = 0.7,
    ) -> None:
        """初始化 InsightFace.

        Args:
            model_name: 模型包名（buffalo_l / antelopev2）.
            det_size: 检测输入边长.
            ctx_id: CUDA 设备号；<0 表示 CPU.
            min_box_size: 人体框最小边长（像素），太小不值得裁人脸.
            det_score: 人脸检测得分门槛.
        """
        # 必须在 import insightface/onnxruntime 之前预加载 CUDA 运行库，
        # 否则 provider .so 解析 libnvrtc/libcudart 失败会静默退回 CPU EP
        from modules.identity.ort_cuda import ensure_onnxruntime_cuda

        if ctx_id is not None and ctx_id >= 0:
            if not ensure_onnxruntime_cuda():
                logger.warning("onnxruntime 未启用 CUDA EP，人脸推理将跑在 CPU 上")

        from insightface.app import FaceAnalysis

        # insightface 0.7.x 的 prepare(ctx_id) 不负责选卡，必须靠 provider_options
        if ctx_id is not None and ctx_id >= 0:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            # cudnn_conv_algo_search 默认 EXHAUSTIVE 会对每个新输入形状重新 benchmark，
            # 人脸裁剪尺寸逐帧变化时会反复搜索、造成秒级抖动，这里固定为 HEURISTIC
            provider_options = [
                {
                    "device_id": int(ctx_id),
                    "cudnn_conv_algo_search": "HEURISTIC",
                },
                {},
            ]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=providers,
            provider_options=provider_options,
        )
        self._app.prepare(ctx_id=int(ctx_id), det_size=(det_size, det_size))
        self.min_box_size = int(min_box_size)
        self.det_score = float(det_score)
        logger.info(
            "InsightFace 就绪: model=%s ctx_id=%s det_size=%d",
            model_name, ctx_id, det_size,
        )

    def extract(
        self, frame: np.ndarray, box: Sequence[float], pad_ratio: float = 0.08
    ) -> Optional[np.ndarray]:
        """从人体框内裁人脸并提取特征.

        Args:
            frame: BGR 原始帧.
            box: 人体框 [x1, y1, x2, y2].
            pad_ratio: 框外扩比例，补回被裁掉的下巴/额头.

        Returns:
            归一化后的特征向量 (D,)；没有合格人脸返回 None.
        """
        if frame is None or box is None or len(box) < 4:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        bw, bh = x2 - x1, y2 - y1
        if bw < self.min_box_size or bh < self.min_box_size:
            return None
        px, py = bw * pad_ratio, bh * pad_ratio
        x1 = max(0, int(x1 - px))
        y1 = max(0, int(y1 - py))
        x2 = min(w, int(x2 + px))
        y2 = min(h, int(y2 + py))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None

        crop = frame[y1:y2, x1:x2]
        try:
            faces = self._app.get(crop)
        except Exception as exc:  # 单帧人脸推理失败不应中断主循环
            logger.debug("人脸推理失败: %s", exc)
            return None
        if not faces:
            return None

        faces = [f for f in faces if float(getattr(f, "det_score", 0.0)) >= self.det_score]
        if not faces:
            return None
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        emb = getattr(face, "embedding", None)
        if emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        if norm < 1e-6:
            return None
        return emb / norm

    @staticmethod
    def stack(templates: List[np.ndarray]) -> Optional[np.ndarray]:
        """把模板列表堆成矩阵.

        Args:
            templates: 特征向量列表.

        Returns:
            (N, D) 矩阵；空列表返回 None.
        """
        if not templates:
            return None
        return np.vstack([np.asarray(t, dtype=np.float32).reshape(1, -1) for t in templates])
