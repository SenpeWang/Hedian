"""核心框架层.

提供所有业务模块共用的基础设施: 消息总线与主题常量、全局配置、
推理流写入端与同步中间件、模块基类、日志系统与路径配置.
"""
from core.event_bus import EventBus, EventTopic
from core.config_manager import ConfigManager
from core.inference_stream import InferenceStream
from core.inference_sync import InferenceSync
from core.base_module import BaseModule
from core.logger import setup_logger, add_root_file_handler
from core.path_manager import PathConfig

__all__ = [
    "EventBus",
    "EventTopic",
    "ConfigManager",
    "InferenceStream",
    "InferenceSync",
    "BaseModule",
    "setup_logger",
    "add_root_file_handler",
    "PathConfig",
]
