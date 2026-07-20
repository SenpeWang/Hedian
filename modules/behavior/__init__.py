"""
行为检测模块

负责：
- 举手检测（供tracker调用）
- 手指屏幕检测

使用方式：
    from modules.behavior import BehaviorModule
    module = BehaviorModule(event_bus, config, paths, display_buffer)
    module.start(video_path, run_id)
"""
from modules.behavior.behavior_module import BehaviorModule
from modules.behavior.hand_raiser import HandRaiser
from modules.behavior.finger_screen_detector import FingerScreenDetector
from modules.behavior.storage_behavior import BehaviorStorage

__all__ = [
    "BehaviorModule",
    "HandRaiser",
    "FingerScreenDetector",
    "BehaviorStorage",
]
