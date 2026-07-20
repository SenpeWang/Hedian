"""
Behavior 结果存储模块（独立文件）

负责保存 Behavior 模块的行为关键事件。
"""
import json
import logging
import os
from typing import List, Dict

from core.path_manager import PathConfig

logger = logging.getLogger("module.behavior.storage")


from core.base_storage import BaseStorage

class BehaviorStorage(BaseStorage):
    """
    Behavior 结果存储器

    负责保存行为关键事件（behavior_key_moments.json）。
    """

    def __init__(self, paths: PathConfig):
        """
        初始化存储器

        Args:
            paths: 路径配置
        """
        super().__init__(paths, "behavior")

    def save_key_moments(self, run_id: str, events: List[Dict]) -> None:
        """
        保存行为关键事件

        空事件列表会跳过保存。

        Args:
            run_id: 运行 ID
            events: 事件列表
        """
        if not events:
            logger.info("没有事件可保存，跳过保存")
            return
        self._save_json_atomic("behavior_key_moments.json", run_id, events)
