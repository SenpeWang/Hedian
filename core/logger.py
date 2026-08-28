"""统一日志系统.

所有模块经本模块创建 logger, 保证日志格式一致. root logger 只初始化一次;
子进程可自行追加文件 handler, 或把 root 的 FileHandler 重定向到当前 session 目录.
"""

import logging
import os
import sys
from pathlib import Path

_FMT = "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s"
_DATEFMT = "%H:%M:%S"

# root logger 是否已初始化(全局只配一次, 避免重复挂载 handler 导致日志重复输出)
_ROOT_CONFIGURED = False


def setup_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """初始化 root logger 后返回指定名称的 logger.

    首次调用时按 level 挂载 stdout handler; 之后的调用只做名称路由, 不重复配置.

    Args:
        name: logger 名称; None 时返回 root logger.
        level: root logger 的日志级别, 默认 INFO.

    Returns:
        指定名称的 logger 实例.
    """
    global _ROOT_CONFIGURED

    if not _ROOT_CONFIGURED:
        root = logging.getLogger()
        root.setLevel(level)
        if not root.handlers:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(level)
            console.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
            root.addHandler(console)
        _ROOT_CONFIGURED = True

    return logging.getLogger(name)


def add_root_file_handler(log_file: str, level: int = logging.INFO) -> None:
    """向 root logger 追加文件 handler.

    多进程场景下每个子进程各自调用一次, 使日志同时写控制台与文件.

    Args:
        log_file: 日志文件路径, 父目录不存在时自动创建.
        level: handler 的日志级别, 默认 INFO.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    logging.getLogger().addHandler(file_handler)


def redirect_file_logger(new_log_file_path: str) -> None:
    """把 root logger 的 FileHandler 重定向到指定日志文件.

    先移除并关闭现有 FileHandler 再挂载新的, 避免多个 handler 同时写不同文件;
    传入空路径时只清理不新增.

    Args:
        new_log_file_path: 新的日志文件路径; 空字符串则仅移除现有 handler.
    """
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    if new_log_file_path:
        os.makedirs(os.path.dirname(new_log_file_path), exist_ok=True)
        handler = logging.FileHandler(new_log_file_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        root_logger.addHandler(handler)
