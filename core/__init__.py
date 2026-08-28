"""
核心框架层.

提供所有业务模块共用的基础设施：
- event_bus: 事件流 (EventBus)
- config_manager: 配置管理
- inference_stream: 推理流写入端 (InferenceStream)
- inference_sync: 推理同步中间件 (InferenceSync)
- base_module: 模块基类
- logger: 日志系统
- path_manager: 路径管理
"""
from core.event_bus import EventBus, EventTopic
from core.config_manager import ConfigManager
from core.inference_stream import InferenceStream
from core.inference_sync import InferenceSync
from core.base_module import BaseModule
from core.logger import setup_logger, add_root_file_handler
from core.path_manager import PathManager

__all__ = [
    "EventBus",
    "EventTopic",
    "ConfigManager",
    "InferenceStream",
    "InferenceSync",
    "BaseModule",
    "setup_logger",
    "add_root_file_handler",
    "PathManager",
]
