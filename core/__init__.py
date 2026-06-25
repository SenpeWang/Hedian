"""
核心框架层

提供所有业务模块共用的基础设施：
- message_bus: 消息总线
- config_manager: 配置管理
- frontend_sync: 前端同步
- base_module: 模块基类
- logger: 日志系统
- path_manager: 路径管理
"""
from core.message_bus import MessageBus, MsgType
from core.config_manager import ConfigManager
from core.frontend_sync import FrontendSync
from core.base_module import BaseModule
from core.logger import setup_logger, get_module_logger
from core.path_manager import PathConfig

__all__ = [
    "MessageBus",
    "MsgType",
    "ConfigManager",
    "FrontendSync",
    "BaseModule",
    "setup_logger",
    "get_module_logger",
    "PathConfig",
]
