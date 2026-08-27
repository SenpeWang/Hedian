"""结果存储抽象基类."""
import os
import json
import logging
from typing import Any
from core.path_manager import PathManager

logger = logging.getLogger("core.storage")


class BaseStorage:
    """结果存储抽象基类，封装通用的原子性 JSON 保存逻辑."""

    def __init__(self, paths: PathManager, module_name: str):
        """初始化."""
        self._paths = paths
        self._module_name = module_name

    def _save_json_atomic(self,
                          filename: str,
                          run_id: str,
                          data: Any,
                          indent: int = 2,
                          output_path: str = None) -> bool:
        """原子性地将数据保存为 JSON 文件.

        Args:
            filename (str): 结果文件名 (output_path 未提供时按此解析路径).
            run_id (str): 本次运行标识.
            data (Any): 待序列化的数据.
            indent (int): JSON 缩进, 默认 2.
            output_path (str): 显式输出路径; 提供时跳过路径解析(调用方已持锁解析过).

        Returns:
            bool: 保存成功返回 True, 失败记录日志后返回 False.
        """
        if output_path is None:
            output_path = str(self._paths.get_result_path(
                run_id=run_id,
                module=self._module_name,
                filename=filename,
            ))
        try:
            tmp_path = str(output_path) + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            os.replace(tmp_path, str(output_path))
            logger.info(f"保存数据到 {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存 {filename} 失败: {e}", exc_info=True)
            return False
