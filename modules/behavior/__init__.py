"""
行为检测模块

负责：
- 手指屏幕检测

使用方式：
    from modules.behavior import BehaviorModule
    module = BehaviorModule(bus, config, paths, aggregator)
    module.start(video_path, run_id)
"""
from modules.behavior.behavior_module import BehaviorModule
from modules.behavior.finger_screen_detector import FingerScreenDetector

__all__ = [
    "BehaviorModule",
    "FingerScreenDetector",
]
