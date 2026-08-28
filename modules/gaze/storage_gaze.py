"""Gaze 结果存储：保存注视告警关键事件."""
import logging
from typing import List, Dict

from core.path_manager import PathConfig

logger = logging.getLogger("module.gaze.storage")

from core.base_storage import BaseStorage


class GazeStorage(BaseStorage):
    """保存凝视告警关键事件（gaze_key_moments.json."""

    def __init__(self, paths: PathConfig):
        """初始化."""
        super().__init__(paths, "gaze")

    def save_key_moments(self, run_id: str, events: List[Dict]) -> None:
        """保存keymoments."""
        if not events:
            logger.info("没有注视告警事件，跳过保存")
            return
        self._save_json_atomic("gaze_key_moments.json", run_id, events)
