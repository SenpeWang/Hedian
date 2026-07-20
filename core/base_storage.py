"""
结果存储抽象基类
"""
import os
import json
import logging
from typing import List, Dict
from core.path_manager import PathConfig

logger = logging.getLogger("core.storage")

class BaseStorage:
    """结果存储抽象基类，封装通用的原子性 JSON 保存逻辑"""

    def __init__(self, paths: PathConfig, module_name: str):
        self._paths = paths
        self._module_name = module_name

    def _save_json_atomic(self, filename: str, run_id: str, data: any, indent: int = 2) -> None:
        """原子性地将数据保存为 JSON 文件"""
        output_path = self._paths.get_result_path(
            run_id=run_id,
            module=self._module_name,
            filename=filename,
        )
        try:
            tmp_path = str(output_path) + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            os.replace(tmp_path, str(output_path))
            logger.info(f"保存数据到 {output_path}")
        except Exception as e:
            logger.error(f"保存 {filename} 失败: {e}", exc_info=True)
