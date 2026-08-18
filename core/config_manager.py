"""
全局配置管理.

从 config.yaml 加载配置，提供统一访问接口。
所有硬编码值集中在此管理。
"""
import os
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger("core.config_manager")

# 基础目录
BASE_DIR = Path(__file__).parent.parent

# 默认配置
_DEFAULTS = {
    "app": {
        "gpu": "0",
        "gpu_map": {"tracker": "0", "voice": "1", "behavior": "1", "evaluation": "1"},
        "gpu_default": "0",
        "fps": 30.0,
        "port": 5002,
    },
    "paths": {
        "data_root": "data",
        "model_root": "models",
        "result_root": "data/results",
    },
    "videos": {
        "front": "data/videos/camFRONT.mpg",
        "pop": "data/videos/camPOP.mpg",
    },
    "supervision": {
        "bind_hold_sec": 10.0,
        "unbind_hold_sec": 20.0,
        "dist_close_px": 200,
        "dist_near_px": 560,
        "consec_raise": 3,
        "consec_idle": 3,
    },
    "event_bus": {
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
    "behavior": {
        "finger_screen": {
            "detect_conf": 0.3,
            "pose_conf": 0.5,
            "hand_to_screen_dist": 400,
            "cooldown_sec": 1.5,
        },
        "finger_file": {
            "detect_conf": 0.25,
            "track_iou": 0.5,
            "file_iou_threshold": 0.2,
            "cooldown_sec": 1.5,
        },
    },
    "modules": {
        "voice": True,
        "tracker": True,
        "gaze": True,
        "behavior": False,
    },
    "rules": {
        "supervision": True,
        "self_ticket": True,
        "personnel_status": True,
        "info_notice": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """deep合并."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class ConfigManager:
    """
    全局配置管理器（单例）.

    从 config.yaml 加载配置，提供统一访问接口。
    """

    _instance: Optional["ConfigManager"] = None

    def __init__(self, config_path: str = None):
        """初始化."""
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
        """加载."""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置."""
        cls._instance = None

    @property
    def gpu(self) -> str:
        """返回 GPU 设备编号."""
        return self._data["app"].get("gpu_default") or self._data["app"].get("gpu", "0")

    @property
    def gpu_map(self) -> dict:
        """获取视角级 GPU 映射 {module_name: gpu_str}."""
        return self._data["app"].get("gpu_map", {})

    @property
    def gpu_default(self) -> str:
        """获取默认 GPU 编号(模块未在 gpu_map 时回退)."""
        return self._data["app"].get("gpu_default") or self._data["app"].get("gpu", "0")

    @property
    def fps(self) -> float:
        """返回推理帧率."""
        return self._data["app"]["fps"]

    @property
    def data_root(self) -> str:
        """返回数据根目录."""
        return self._data["paths"]["data_root"]

    @property
    def model_root(self) -> str:
        """返回模型根目录."""
        return self._data["paths"]["model_root"]

    @property
    def result_root(self) -> str:
        """返回结果根目录."""
        return self._data["paths"]["result_root"]

    @property
    def video_path(self) -> str:
        """返回默认视频路径."""
        return self._data["videos"]["front"]

    @property
    def supervision(self) -> dict:
        """返回监护相关配置."""
        return self._data["supervision"]

    @property
    def event_bus(self) -> dict:
        """返回事件总线配置."""
        return self._data["event_bus"]

    @property
    def voice(self) -> dict:
        """返回语音相关配置."""
        return self._data["voice"]

    @property
    def gaze(self) -> dict:
        """返回注视相关配置."""
        return self._data["gaze"]

    @property
    def behavior(self) -> dict:
        """返回行为相关配置."""
        return self._data["behavior"]

    @property
    def videos(self) -> dict:
        """返回视频路径配置."""
        return self._data["videos"]

    @property
    def modules(self) -> dict:
        """返回各模块启用开关."""
        return self._data["modules"]

    @property
    def rules(self) -> dict:
        """返回各制度启用开关."""
        return self._data["rules"]

    def get(self, key: str, default: Any = None) -> Any:
        """获取."""
        # key 支持点号分隔，如 'app.gpu'
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def to_dict(self) -> dict:
        """转换为dict."""
        return dict(self._data)
