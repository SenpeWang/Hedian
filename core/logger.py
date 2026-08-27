"""
统一日志系统.

所有模块使用此模块创建 logger，保证日志格式统一。
"""
import os
import logging
import sys
from pathlib import Path

_FMT = "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s"
_DATEFMT = "%H:%M:%S"

_root_configured = False


def setup_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """配置并返回 logger."""
    global _root_configured

    if not _root_configured:
        root = logging.getLogger()
        root.setLevel(level)
        if not root.handlers:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(level)
            console.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
            root.addHandler(console)
        _root_configured = True

    return logging.getLogger(name)


def add_root_file_handler(log_file: str, level: int = logging.INFO) -> None:
    """向 root logger 添加文件 handler，用于多进程场景."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    logging.getLogger().addHandler(file_handler)


def redirect_file_logger(new_log_file_path: str) -> None:
    """重定向根日志 FileHandler 到当前活跃 session 目录."""
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
