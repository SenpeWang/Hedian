"""
核心框架层

提供所有业务模块共用的基础设施：
- event_bus: 消息总线
- config_manager: 配置管理
- display_buffer: 前端同步
- base_module: 模块基类
- logger: 日志系统
- path_manager: 路径管理
"""
from core.event_bus import EventBus, EventTopic
from core.config_manager import ConfigManager
from core.display_buffer import DisplayBuffer
from core.inference_bus import InferenceBus
from core.module_sync import ModuleSync
from core.base_module import BaseModule
from core.logger import setup_logger, get_module_logger
from core.path_manager import PathConfig

__all__ = [
    "EventBus",
    "EventTopic",
    "ConfigManager",
    "DisplayBuffer",
    "InferenceBus",
    "ModuleSync",
    "BaseModule",
    "setup_logger",
    "get_module_logger",
    "PathConfig",
]
