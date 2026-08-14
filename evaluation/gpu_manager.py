"""智能动态 GPU 资源探测与调度管理模块.

负责多卡计算环境下 GPU 状态（显存剩余、算力利用率）的实时探测、
智能加权优选与大模型推理子进程的运行环境配置。
"""
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger("evaluation.gpu_manager")


class GPUManager:
    """智能 GPU 资源探测与调度管理器.

    提供基于 NVML 与 nvidia-smi 的双重多维探测能力，
    依据空闲显存量与算力负载加权优选最适合大模型加载的物理 GPU。
    """

    @staticmethod
    def get_gpu_stats() -> List[Dict[str, Any]]:
        """实时探测主机中所有物理 GPU 的显存与利用率状态.

        优先采用 pynvml (NVML C-API) 驱动接口读取；
        若 NVML 不可用，自动降级执行 nvidia-smi 命令行解析。

        Returns:
            List[Dict[str, Any]]: 包含各 GPU 状态的列表，每个字典包含：
                - 'index' (int): 物理 GPU 索引编号
                - 'name' (str): 显卡型号名称
                - 'total_memory_mb' (float): 总显存（MB）
                - 'free_memory_mb' (float): 剩余空闲显存（MB）
                - 'used_memory_mb' (float): 已占用显存（MB）
                - 'gpu_utilization_pct' (float): GPU 算力利用率百分比 (0-100)
                - 'memory_utilization_pct' (float): 显存利用率百分比 (0-100)
        """
        gpu_stats_list: List[Dict[str, Any]] = []

        # 方法一：优先使用 pynvml
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for gpu_idx in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                device_name = pynvml.nvmlDeviceGetName(handle)

                gpu_stats_list.append({
                    "index": gpu_idx,
                    "name": str(device_name),
                    "total_memory_mb": float(memory_info.total / (1024 ** 2)),
                    "free_memory_mb": float(memory_info.free / (1024 ** 2)),
                    "used_memory_mb": float(memory_info.used / (1024 ** 2)),
                    "gpu_utilization_pct": float(utilization_info.gpu),
                    "memory_utilization_pct": float(utilization_info.memory),
                })
            pynvml.nvmlShutdown()
            if gpu_stats_list:
                return gpu_stats_list
        except Exception as pynvml_error:
            logger.debug(f"pynvml 探测失败，尝试切换 nvidia-smi 命令行: {pynvml_error}")

        # 方法二：降级使用 nvidia-smi 命令行
        try:
            command = [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ]
            process_output = subprocess.check_output(
                command, stderr=subprocess.DEVNULL, timeout=5
            ).decode("utf-8")

            for line in process_output.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 7:
                    gpu_stats_list.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "total_memory_mb": float(parts[2]),
                        "free_memory_mb": float(parts[3]),
                        "used_memory_mb": float(parts[4]),
                        "gpu_utilization_pct": float(parts[5]),
                        "memory_utilization_pct": float(parts[6]),
                    })
            if gpu_stats_list:
                return gpu_stats_list
        except Exception as cli_error:
            logger.warning(f"nvidia-smi 命令行探测失败: {cli_error}")

        return gpu_stats_list

    @classmethod
    def select_best_gpu(cls, min_free_memory_mb: float = 10000.0) -> int:
        """依据剩余显存与空闲算力综合加权评选最佳物理 GPU.

        评分公式：
            Score = (Free_VRAM / Total_VRAM) * 0.70 + (1.0 - GPU_Util / 100.0) * 0.30

        Args:
            min_free_memory_mb (float): 模型加载所需的最低空闲显存（MB），默认 10000.0 MB.

        Returns:
            int: 选中的物理 GPU 索引编号；未探测到可用 GPU 时安全回退为 0.
        """
        gpu_stats_list = cls.get_gpu_stats()
        if not gpu_stats_list:
            logger.warning("未探测到任何 GPU，回退为默认 GPU 0")
            return 0

        # 过滤满足显存最低门槛的候选卡
        candidate_gpus = [
            gpu_info for gpu_info in gpu_stats_list
            if gpu_info["free_memory_mb"] >= min_free_memory_mb
        ]

        # 若均未达到硬性门槛，则退化为在全部可用卡中选择剩余显存最大者
        if not candidate_gpus:
            logger.warning(
                f"所有 GPU 空闲显存均未达到门槛 ({min_free_memory_mb:.1f} MB)，"
                "退化为选取剩余显存最大的 GPU"
            )
            candidate_gpus = gpu_stats_list

        best_gpu_index: int = 0
        highest_score: float = -1.0

        for gpu_info in candidate_gpus:
            free_ratio = (
                gpu_info["free_memory_mb"] / max(1.0, gpu_info["total_memory_mb"])
            )
            idle_ratio = 1.0 - (gpu_info["gpu_utilization_pct"] / 100.0)
            composite_score = free_ratio * 0.70 + idle_ratio * 0.30

            logger.info(
                f"GPU {gpu_info['index']} ({gpu_info['name']}): "
                f"空闲显存={gpu_info['free_memory_mb']:.1f}MB, "
                f"利用率={gpu_info['gpu_utilization_pct']}%, "
                f"综合得分={composite_score:.4f}"
            )

            if composite_score > highest_score:
                highest_score = composite_score
                best_gpu_index = gpu_info["index"]

        logger.info(f"智能动态 GPU 调度决选: 物理 GPU {best_gpu_index} (最高得分 {highest_score:.4f})")
        return best_gpu_index

    @staticmethod
    def setup_gpu_environment(gpu_index: int) -> None:
        """在子进程中配置 GPU 环境变量与 CPU 线程限制.

        Args:
            gpu_index (int): 选中的物理 GPU 索引编号.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
