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
    """
    创建模块专用 logger

    同时确保 root logger 也有 handler，让所有子 logger 继承。

    Args:
        name: logger 名称（通常用模块名）
        log_file: 日志文件路径（可选）
        level: 日志级别
        format_str: 自定义格式（可选）

    Returns:
        配置好的 logger
    """
    global _root_configured

    # 默认格式
    if format_str is None:
        format_str = "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s"

    formatter = logging.Formatter(format_str, datefmt="%H:%M:%S")

    # 配置 root logger，让所有子模块的 logger 都能输出
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

    # 文件输出（可选）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_module_logger(module_name: str, result_dir: str = None) -> logging.Logger:
    """
    获取模块 logger

    Args:
        module_name: 模块名称（如 'voice', 'tracker', 'gaze'）
        result_dir: 结果目录（用于日志文件）

    Returns:
        配置好的 logger
    """
    log_file = None
    if result_dir:
        log_file = str(Path(result_dir) / f"{module_name}.log")

    return setup_logger(f"module.{module_name}", log_file=log_file)
