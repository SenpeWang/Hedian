"""让 onnxruntime 的 CUDA EP 在本环境可用的预加载补丁.

问题背景：本机 `ldconfig` 缓存里指向 `/usr/local/cuda-12.8/...`，但该目录实际不存在，
导致 `libonnxruntime_providers_cuda.so` 解析 `libnvrtc.so.12` / `libcudart.so.12` 失败、
悄悄退回 CPU EP（人脸推理慢数倍）。真正的 CUDA 运行库在
`site-packages/nvidia/*/lib`（torch 的 pip 分发包）。

不能在 import 之后再设 `LD_LIBRARY_PATH`（glibc 在进程启动时就读走了），
所以这里用 `ctypes.CDLL(..., RTLD_GLOBAL)` 按绝对路径预加载：
一旦同名 SONAME 已进进程，后续 dlopen 依赖会直接复用已加载对象。

注意：Hedian 的 `_run_module_process` 会主动清空 `LD_LIBRARY_PATH`（防 torch 误链接），
因此该补丁只能在此处做进程内预加载，不能依赖环境变量。
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
from typing import Set

logger = logging.getLogger("module.identity.ort_cuda")

_LOADED: Set[str] = set()


def preload_cuda_libs() -> int:
    """把 site-packages/nvidia/*/lib 下的 CUDA 运行库预加载进进程.

    Returns:
        成功加载的库数量；没有 nvidia 包或全部失败时返回 0.
    """
    try:
        import nvidia  # noqa: F401  (仅用于定位 site-packages 路径)
    except ImportError:
        logger.warning("未找到 nvidia pip 包，跳过 CUDA 库预加载")
        return 0

    base = os.path.dirname(nvidia.__file__)
    lib_paths = []
    for lib_dir in sorted(glob.glob(os.path.join(base, "*", "lib"))):
        lib_paths.extend(sorted(glob.glob(os.path.join(lib_dir, "*.so*"))))
    if not lib_paths:
        logger.warning("nvidia 包目录下没有 .so，跳过预加载")
        return 0

    # 依赖有先后关系，多轮扫描：每轮只加载「本轮能加载成功」的，直到不再有进展
    loaded = 0
    for _ in range(4):
        progressed = False
        for path in lib_paths:
            if path in _LOADED:
                continue
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            _LOADED.add(path)
            loaded += 1
            progressed = True
        if not progressed:
            break

    logger.info("CUDA 运行库预加载完成：%d/%d", loaded, len(lib_paths))
    return loaded


def ensure_onnxruntime_cuda() -> bool:
    """预加载 CUDA 库并检查 onnxruntime 的 CUDA EP 是否真的可用.

    必须在 import onnxruntime / insightface 之前调用（本函数内部才 import ort）。

    Returns:
        CUDAExecutionProvider 可用返回 True.
    """
    if not _LOADED:
        preload_cuda_libs()
    try:
        import onnxruntime as ort
    except Exception as exc:
        logger.error("导入 onnxruntime 失败: %s", exc)
        return False
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        logger.info("onnxruntime CUDA EP 可用: %s", providers)
        return True
    logger.warning("onnxruntime 仍只有 CPU EP: %s", providers)
    return False
