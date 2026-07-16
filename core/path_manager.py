"""
统一路径管理

所有路径通过此模块管理，避免硬编码。
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PathConfig:
    """
    路径配置

    所有路径都基于 base_dir 计算，避免硬编码。
    """
    base_dir: Path
    data_root: Path
    model_root: Path
    result_root: Path

    @classmethod
    def from_config(cls, config: dict, base_dir: str = None) -> "PathConfig":
        """
        从配置字典创建 PathConfig

        Args:
            config: 配置字典
            base_dir: 基础目录（默认使用当前文件所在目录的父目录）

        Returns:
            PathConfig 实例
        """
        if base_dir is None:
            base_dir = str(Path(__file__).parent.parent)

        base = Path(base_dir)
        paths_config = config.get("paths", {})

        return cls(
            base_dir=base,
            data_root=base / paths_config.get("data_root", "data"),
            model_root=base / paths_config.get("model_root", "models"),
            result_root=base / paths_config.get("result_root", "data/results"),
        )

    def get_model_path(self, category: str, filename: str) -> Path:
        """
        获取模型文件路径

        Args:
            category: 模型类别（如 'detection', 'gaze', 'behavior'）
            filename: 模型文件名

        Returns:
            模型文件完整路径
        """
        return self.model_root / category / filename

    def get_video_path(self, filename: str) -> Path:
        """
        获取视频文件路径

        Args:
            filename: 视频文件名

        Returns:
            视频文件完整路径
        """
        return self.data_root / "videos" / filename

    def get_result_path(self, run_id: str, module: str, filename: str) -> Path:
        """
        获取结果文件路径（自动创建目录）

        Args:
            run_id: 运行 ID（如 '20260617_120000'）
            module: 模块名称（如 'voice', 'tracker', 'gaze'）
            filename: 结果文件名

        Returns:
            结果文件完整路径
        """
        path = self.result_root / run_id / module
        path.mkdir(parents=True, exist_ok=True)
        return path / filename

    def get_result_dir(self, run_id: str, module: str = None) -> Path:
        """
        获取结果目录（自动创建）

        Args:
            run_id: 运行 ID
            module: 模块名称（可选）

        Returns:
            结果目录路径
        """
        if module:
            path = self.result_root / run_id / module
        else:
            path = self.result_root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_dirs(self) -> None:
        """确保所有必要的目录存在"""
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.result_root.mkdir(parents=True, exist_ok=True)

        # 创建模型子目录
        for category in ("detection", "gaze", "behavior", "evaluation"):
            (self.model_root / category).mkdir(parents=True, exist_ok=True)

        # 创建数据子目录
        (self.data_root / "videos").mkdir(parents=True, exist_ok=True)
