"""结果存储抽象基类.

封装通用的原子性 JSON 保存逻辑, 供各业务模块的结果存储器继承.
"""
import json
import logging
import os
from typing import Any, Optional

from core.path_manager import PathConfig

logger = logging.getLogger("core.storage")


class BaseStorage:
    """结果存储抽象基类, 封装通用的原子性 JSON 保存逻辑."""

    def __init__(self, paths: PathConfig, module_name: str):
        """初始化.

        Args:
            paths: 路径配置, 用于解析结果输出目录.
            module_name: 产出模块名, 结果按其分目录.
        """
        self._paths = paths
        self._module_name = module_name

    def _save_json_atomic(
        self,
        filename: str,
        run_id: str,
        data: Any,
        indent: int = 2,
        output_path: Optional[str] = None,
    ) -> bool:
        """原子性地将数据保存为 JSON 文件.

        先写同名 .tmp 文件再 os.replace, 保证读者不会读到写一半的文件.

        Args:
            filename: 结果文件名(output_path 未提供时按此解析路径).
            run_id: 本次运行标识.
            data: 待序列化的数据.
            indent: JSON 缩进, 默认 2.
            output_path: 显式输出路径; 提供时跳过路径解析(调用方已持锁解析过).

        Returns:
            保存成功返回 True, 失败记录日志后返回 False.
        """
        if output_path is None:
            output_path = str(self._paths.get_result_path(
                run_id=run_id,
                module=self._module_name,
                filename=filename,
            ))
        try:
            temp_path = str(output_path) + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            os.replace(temp_path, str(output_path))
            logger.info(f"保存数据到 {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存 {filename} 失败: {e}", exc_info=True)
            return False
