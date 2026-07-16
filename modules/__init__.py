"""
业务模块层

包含所有业务模块：
- voice: 语音模块（语音转文字 + 意图分类）
- tracker: 多目标跟踪模块（目标检测 + 跟踪 + 举手检测）
- gaze: 注视检测模块（头部检测 + 注视推断）
- behavior: 行为检测模块（手指屏幕检测）

所有模块继承 BaseModule，实现统一接口。
"""
from modules.voice.voice_module import VoiceModule
from modules.tracker.tracker_module import TrackerModule
from modules.behavior.behavior_module import BehaviorModule

__all__ = [
    "VoiceModule",
    "TrackerModule",
    "BehaviorModule",
]
