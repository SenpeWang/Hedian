"""
全局配置管理

从 config.yaml 加载配置，提供统一访问接口。
所有硬编码值集中在此管理。
"""
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger("core.config_manager")

# 基础目录
BASE_DIR = Path(__file__).parent.parent

# 默认配置
_DEFAULTS = {
    "app": {
        "gpu": "0",
        "fps": 30.0,
    },
    "paths": {
        "data_root": "data",
        "model_root": "models",
        "result_root": "data/results",
        "video": "data/videos/camFRONT.mpg",
    },
    "supervision": {
        "bind_hold_sec": 3.0,
        "unbind_hold_sec": 20.0,
        "dist_close_px": 280,
        "dist_near_px": 560,
        "vote_window": 3,
        "vote_threshold": 2,
        "cooldown_frames": 300,
    },
    "bus": {
        "max_queue_size": 1024,
    },
    "voice": {
        "sample_rate": 16000,
        "sentence_gap_sec": 1.5,
        "device_pattern": r"(1ES\w+|T1RPA\w+|LCO\w+|RPA\w+|SM3)",
        "action_verbs": ["开启", "关闭", "长按", "调出", "停运"],
        "confirm_words": ["好", "确认", "没问题", "收到", "明白", "正确"],
        "verify_words": ["核对", "核实", "验证", "检查"],
    },
    "gaze": {
        "head_conf_th": 0.55,
        "inout_th": 0.5,
        "heatmap_th": 0.3,
        "head_min_size": 20,
        "head_max_size": 300,
    },
    "modules": {
        "voice": True,
        "mot": True,
        "gaze": True,
        "behavior": False,
    },
    "regulations": {
        "supervision": True,
        "self_ticket": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """
    深度合并 override 到 base

    Args:
        base: 基础配置
        override: 覆盖配置

    Returns:
        合并后的配置
    """
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class ConfigManager:
    """
    全局配置管理器（单例）

    从 config.yaml 加载配置，提供统一访问接口。
    """

    _instance: Optional["ConfigManager"] = None

    def __init__(self, config_path: str = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径（默认使用 config.yaml）
        """
        self._data = dict(_DEFAULTS)
        yaml_path = config_path or str(BASE_DIR / "config.yaml")

        if os.path.exists(yaml_path):
            with open(yaml_path, encoding="utf-8") as f:
                user = yaml.safe_load(f) or {}
            self._data = _deep_merge(self._data, user)
            logger.info(f"加载配置: {yaml_path}")
        else:
            logger.warning(f"配置文件不存在: {yaml_path}，使用默认配置")

    @classmethod
    def load(cls, config_path: str = None) -> "ConfigManager":
        """
        获取配置管理器实例（单例）

        Args:
            config_path: 配置文件路径

        Returns:
            ConfigManager 实例
        """
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置单例"""
        cls._instance = None

    @property
    def gpu(self) -> str:
        """GPU 编号"""
        return self._data["app"]["gpu"]

    @property
    def fps(self) -> float:
        """帧率"""
        return self._data["app"]["fps"]

    @property
    def data_root(self) -> str:
        """数据根目录"""
        return str(BASE_DIR / self._data["paths"]["data_root"])

    @property
    def model_root(self) -> str:
        """模型根目录"""
        return str(BASE_DIR / self._data["paths"]["model_root"])

    @property
    def result_root(self) -> str:
        """结果根目录"""
        return str(BASE_DIR / self._data["paths"]["result_root"])

    @property
    def video_path(self) -> str:
        """视频路径"""
        return str(BASE_DIR / self._data["paths"]["video"])

    @property
    def supervision(self) -> dict:
        """监护配置"""
        return self._data["supervision"]

    @property
    def bus(self) -> dict:
        """总线配置"""
        return self._data["bus"]

    @property
    def voice(self) -> dict:
        """语音配置"""
        return self._data["voice"]

    @property
    def gaze(self) -> dict:
        """注视配置"""
        return self._data["gaze"]

    @property
    def modules(self) -> dict:
        """模块开关"""
        return self._data["modules"]

    @property
    def regulations(self) -> dict:
        """制度配置"""
        return self._data["regulations"]

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键（支持点号分隔，如 'app.gpu'）
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def to_dict(self) -> dict:
        """
        导出配置为字典

        Returns:
            配置字典
        """
        return dict(self._data)
