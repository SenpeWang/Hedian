"""
统一路径管理.

所有路径通过此模块管理，避免硬编码。
"""
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PathManager:
    """
    路径配置.

    所有路径都基于 base_dir 计算，避免硬编码。
    """

    base_dir: Path
    data_root: Path
    model_root: Path
    result_root: Path

    @classmethod
    def from_config(cls, config: dict, base_dir: str = None) -> "PathManager":
        """根据配置构造 PathManager."""
        if base_dir is None:
            base_dir = str(Path(__file__).parent.parent)

        base = Path(base_dir)
        paths_config = config.get("paths", {})

        # 路径以相对形式存储（相对项目根，main.py 启动时已 os.chdir 到 base_dir）
        return cls(
            base_dir=base,
            data_root=Path(paths_config.get("data_root", "data")),
            model_root=Path(paths_config.get("model_root", "models")),
            result_root=Path(paths_config.get("result_root", "data/results")),
        )

    def get_model_path(self, category: str, filename: str) -> Path:
        """获取model路径."""
        return self.model_root / category / filename

    def get_video_path(self, filename: str) -> Path:
        """获取视频路径."""
        return self.data_root / "videos" / filename

    def get_result_path(self, run_id: str, module: str, filename: str) -> Path:
        """获取结果路径."""
        path = self.result_root / run_id / module
        path.mkdir(parents=True, exist_ok=True)
        return path / filename

    def get_result_dir(self, run_id: str, module: str = None) -> Path:
        """获取结果dir."""
        if module:
            path = self.result_root / run_id / module
        else:
            path = self.result_root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path
