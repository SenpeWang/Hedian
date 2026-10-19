"""全局配置管理.

config.yaml 是唯一配置事实源; 本模块加载该文件并提供统一访问接口.

加载策略(配置缺失时不阻断启动):
- config.yaml 存在 -> 正常加载
- config.yaml 缺失 -> 记录 warning 并以空配置继续, 不抛错;
  各属性以 .get 链兜底返回程序级常量, 避免 KeyError.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("core.config_manager")

# 项目基础目录
BASE_DIR = Path(__file__).parent.parent

# 唯一配置事实源文件名(程序限定加载本文件)
CONFIG_FILENAME = "config.yaml"


class ConfigManager:
    """全局配置管理器: 唯一事实源为项目根下的 config.yaml.

    属性访问用 .get 链兜底, 所带默认值(如 fps=30.0)是防止 KeyError 的程序级常量.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """加载 config.yaml.

        Args:
            config_path: config.yaml 路径;
                None 时取项目根下 CONFIG_FILENAME.
        """
        yaml_path = config_path or str(BASE_DIR / CONFIG_FILENAME)
        if os.path.exists(yaml_path):
            with open(yaml_path, encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"加载配置: {yaml_path}")
        else:
            logger.warning(
                f"配置文件不存在: {yaml_path}，使用空配置(程序级兜底生效)"
            )
            self._config = {}

    @property
    def gpu_default(self) -> str:
        """默认 GPU 编号(命令行未指定 --gpu 时使用).

        Returns:
            GPU 编号字符串, 默认 "0".
        """
        return self._config.get("app", {}).get("gpu_default", "0")

    @property
    def gpu(self) -> str:
        """默认 GPU 编号(gpu_default 优先, 缺省退回 gpu).

        Returns:
            GPU 编号字符串, 默认 "0".
        """
        app = self._config.get("app", {})
        return app.get("gpu_default") or app.get("gpu", "0")

    @property
    def gpu_map(self) -> dict:
        """视角级 GPU 映射 {module_name: gpu_str}.

        Returns:
            模块名到 GPU 编号的映射; 未配置为空字典.
        """
        return self._config.get("app", {}).get("gpu_map", {})

    @property
    def fps(self) -> float:
        """推理帧率.

        Returns:
            帧率(帧/秒), 默认 30.0.
        """
        return self._config.get("app", {}).get("fps", 30.0)

    @property
    def video_path(self) -> str:
        """默认 front 视频路径.

        Returns:
            front 视频路径字符串, 未配置为空字符串.
        """
        return self._config.get("videos", {}).get("front", "")

    def to_dict(self) -> dict:
        """返回配置的浅拷贝.

        Returns:
            配置的浅拷贝字典.
        """
        return dict(self._config)
