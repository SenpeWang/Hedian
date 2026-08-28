"""统一路径配置.

所有路径由本模块基于 base_dir 计算, 避免在业务代码中硬编码.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PathConfig:
    """路径配置.

    所有路径都基于 base_dir 计算, 避免硬编码.

    Attributes:
        base_dir: 项目根目录(main.py 启动时已 os.chdir 到此目录).
        data_root: 数据根目录, 视频与其子目录均在此下.
        model_root: 模型根目录, 各模块权重按子目录区分.
        result_root: 结果根目录, 运行产物按 run_id / module 分目录.
    """

    base_dir: Path
    data_root: Path
    model_root: Path
    result_root: Path

    @classmethod
    def from_config(cls, config: dict, base_dir: Optional[str] = None) -> "PathConfig":
        """根据配置构造路径配置.

        Args:
            config: 完整配置字典, 读取其中 "paths" 子配置.
            base_dir: 项目根目录; None 时取本模块上一级目录.

        Returns:
            PathConfig: 构造好的路径配置实例.
        """
        if base_dir is None:
            base_dir = str(Path(__file__).parent.parent)

        base = Path(base_dir)
        paths_config = config.get("paths", {})

        # 路径以相对形式存储(相对项目根, main.py 启动时已 os.chdir 到 base_dir)
        return cls(
            base_dir=base,
            data_root=Path(paths_config.get("data_root", "data")),
            model_root=Path(paths_config.get("model_root", "models")),
            result_root=Path(paths_config.get("result_root", "data/results")),
        )

    def get_model_path(self, model_subdir: str, filename: str) -> Path:
        """拼接模型权重路径.

        Args:
            model_subdir: model_root 下的子目录名, 如 "tracker".
            filename: 权重文件名.

        Returns:
            Path: model_root / model_subdir / filename.
        """
        return self.model_root / model_subdir / filename

    def get_video_path(self, filename: str) -> Path:
        """拼接视频路径.

        Args:
            filename: 视频文件名.

        Returns:
            Path: data_root / "videos" / filename.
        """
        return self.data_root / "videos" / filename

    def get_result_path(self, run_id: str, module: str, filename: str) -> Path:
        """拼接结果文件路径, 并创建其父目录.

        Args:
            run_id: 本次运行标识.
            module: 产出模块名.
            filename: 结果文件名.

        Returns:
            Path: result_root / run_id / module / filename.
        """
        path = self.result_root / run_id / module
        path.mkdir(parents=True, exist_ok=True)
        return path / filename

    def get_result_dir(self, run_id: str, module: Optional[str] = None) -> Path:
        """获取结果目录, 并创建目录.

        Args:
            run_id: 本次运行标识.
            module: 产出模块名; None 时返回 run_id 级目录.

        Returns:
            Path: module 给定时为 result_root / run_id / module, 否则为 result_root / run_id.
        """
        if module:
            path = self.result_root / run_id / module
        else:
            path = self.result_root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path
