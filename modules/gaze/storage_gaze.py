"""Gaze 结果存储：保存注视告警关键事件."""
import logging
from typing import Dict, List

from core.base_storage import BaseStorage
from core.path_manager import PathConfig

logger = logging.getLogger("module.gaze.storage")


class GazeStorage(BaseStorage):
    """保存凝视告警关键事件（gaze_key_moments.json）."""

    def __init__(self, paths: PathConfig) -> None:
        """初始化存储。

        Args:
            paths: 路径配置。
        """
        super().__init__(paths, "gaze")

    def save_key_moments(self, run_id: str, events: List[Dict]) -> None:
        """保存关键事件.

        Args:
            run_id: 本次运行标识。
            events: 关键事件列表。
        """
        if not events:
            logger.info("没有注视告警事件，跳过保存")
            return
        self._save_json_atomic("gaze_key_moments.json", run_id, events)
