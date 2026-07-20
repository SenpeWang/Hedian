"""
Tracker 结果存储模块（独立文件）

负责保存 Tracker 模块的处理结果。
"""
import json
import logging
import os
from typing import List, Dict

from core.path_manager import PathConfig

logger = logging.getLogger("module.tracker.storage")


from core.base_storage import BaseStorage

class TrackerStorage(BaseStorage):
    """
    Tracker 结果存储器

    负责保存 Tracker 模块的处理结果，包括：
    - 关键事件（tracker_key_moments.json）
    - 角色信息（tracker_role_info.json）
    - 关键帧（key_frames/）
    """

    def __init__(self, paths: PathConfig):
        """
        初始化存储器

        Args:
            paths: 路径配置
        """
        super().__init__(paths, "tracker")

    def save_key_moments(self, run_id: str, events: List[Dict]) -> None:
        """
        保存关键事件

        空事件列表会跳过保存。

        Args:
            run_id: 运行 ID
            events: 事件列表
        """
        if not events:
            logger.info("没有事件可保存，跳过保存")
            return
        for ev in events:
            if "roles_details" not in ev:
                ev["roles_details"] = {}
        self._save_json_atomic("tracker_key_moments.json", run_id, events)

    def save_role_info(self, run_id: str, roles_info: Dict) -> None:
        """
        保存角色信息

        Args:
            run_id: 运行 ID
            roles_info: 角色信息
        """
        self._save_json_atomic("tracker_role_info.json", run_id, {"roles": roles_info})
