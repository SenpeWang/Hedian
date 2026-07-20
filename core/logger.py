"""
统一日志系统

所有模块使用此模块创建 logger，保证日志格式统一。
"""
import logging
import sys
from pathlib import Path
from typing import Optional

_root_configured = False


def setup_logger(
    name: str = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: str = None,
) -> logging.Logger:
    """创建模块专用 logger"""
    global _root_configured

    # 默认格式
    if format_str is None:
        format_str = "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s"

    formatter = logging.Formatter(format_str, datefmt="%H:%M:%S")

    # 配置 root logger
    if not _root_configured:
        root = logging.getLogger()
        root.setLevel(level)
        if not root.handlers:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(level)
            console.setFormatter(formatter)
            root.addHandler(console)
        _root_configured = True

    logger = logging.getLogger(name)

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_module_logger(module_name: str, result_dir: str = None) -> logging.Logger:
    """获取模块 logger"""
    log_file = None
    if result_dir:
        log_file = str(Path(result_dir) / f"{module_name}.log")

    return setup_logger(f"module.{module_name}", log_file=log_file)


def add_root_file_handler(log_file: str, level: int = logging.INFO) -> None:
    """向 root logger 添加文件 handler，用于多进程场景"""
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
