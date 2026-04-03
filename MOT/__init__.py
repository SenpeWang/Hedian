"""
HeDian Multi-Object Tracking Package
核电站监护制合规检测 — MOT 模块
"""
import os
from dataclasses import dataclass
from typing import Dict, Any

__version__ = "1.0.0"

# ── 基础路径 ──
_HEDIAN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DATA_ROOT = os.path.abspath(os.path.join(_HEDIAN, "..", "Hedian_data"))


@dataclass
class PathConfig:
    input_video: str = os.path.join(_DATA_ROOT, "data", "camFRONT.mpg")
    output_dir: str = os.path.join(_DATA_ROOT, "Result")


# ── 工位坐标 ──
WORKSTATIONS = {
    "LEADER": (1049, 398),
    "ROAD1":  (1563, 494),
    "ROAD2":  (1146, 662),
}

# ── 距离阈值（像素） ──
DIST_CLOSE = 280      # 到位
DIST_NEAR  = 560      # 接近中

FORWARD_SECONDS = 10  # 举手前向检测窗口


class Settings:
    """全局设置（单例）"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        from .detector.detector import DetectionConfig
        from .tracker.tracker import TrackerConfig
        self.detection = DetectionConfig()
        self.tracker = TrackerConfig()
        self.path = PathConfig()
        self._initialized = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection": {
                "model_path": self.detection.model_path,
                "pose_model_path": self.detection.pose_model_path,
                "conf_threshold": self.detection.conf_threshold,
            },
            "tracker": {
                "dist_thresh": self.tracker.dist_thresh,
            },
            "path": {
                "input_video": self.path.input_video,
                "output_dir": self.path.output_dir,
            },
        }


SETTINGS = Settings()
