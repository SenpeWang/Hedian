"""
Gaze 结果存储模块（独立文件）

负责保存 Gaze 模块的注视告警关键事件。
"""
import json
import logging
import os
from typing import List, Dict

from core.path_manager import PathConfig

logger = logging.getLogger("module.gaze.storage")


from core.base_storage import BaseStorage

class GazeStorage(BaseStorage):
    """
    Gaze 结果存储器

    负责保存凝视告警关键事件（gaze_key_moments.json）。
    """

    def __init__(self, paths: PathConfig):
        """
        初始化存储器

        Args:
            paths: 路径配置
        """
        super().__init__(paths, "gaze")

    def save_key_moments(self, run_id: str, events: List[Dict]) -> None:
        """
        保存凝视告警关键事件

        空事件列表会跳过保存。

        Args:
            run_id: 运行 ID
            events: 事件列表
        """
        if not events:
            logger.info("没有注视告警事件，跳过保存")
            return
        self._save_json_atomic("gaze_key_moments.json", run_id, events)
