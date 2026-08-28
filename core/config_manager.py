"""全局配置管理.

从 config.yaml 加载配置, 与内置默认值深度合并后提供统一访问接口.
所有硬编码值集中在此管理.
"""
import logging
import os
from pathlib import Path
from typing import Optional

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
    """深度合并两个配置字典.

    两端的子字典递归合并, 其余键直接以 override 覆盖 base.

    Args:
        base: 基底配置(如内置默认值).
        override: 覆盖配置(如用户 config.yaml).

    Returns:
        合并后的新字典, 不改入参.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ConfigManager:
    """全局配置管理器: 从 config.yaml 加载配置, 提供统一访问接口."""

    def __init__(self, config_path: Optional[str] = None):
        """加载配置.

        Args:
            config_path: config.yaml 路径; None 时取项目根下的 config.yaml.
        """
        self._config = dict(_DEFAULTS)
        yaml_path = config_path or str(BASE_DIR / "config.yaml")

        if os.path.exists(yaml_path):
            with open(yaml_path, encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
            self._config = _deep_merge(self._config, user_config)
            logger.info(f"加载配置: {yaml_path}")
        else:
            logger.warning(f"配置文件不存在: {yaml_path}，使用默认配置")

    @property
    def gpu(self) -> str:
        """返回默认 GPU 编号(gpu_default 优先, 缺省时退回 gpu)."""
        return self._config["app"].get("gpu_default") or self._config["app"].get("gpu", "0")

    @property
    def gpu_map(self) -> dict:
        """获取视角级 GPU 映射 {module_name: gpu_str}."""
        return self._config["app"].get("gpu_map", {})

    @property
    def fps(self) -> float:
        """返回推理帧率."""
        return self._config["app"]["fps"]

    @property
    def video_path(self) -> str:
        """返回默认视频路径."""
        return self._config["videos"]["front"]

    def to_dict(self) -> dict:
        """转换为字典.

        Returns:
            配置的浅拷贝.
        """
        return dict(self._config)
