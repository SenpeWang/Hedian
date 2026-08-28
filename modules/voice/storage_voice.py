"""语音结果存储：关键事件（voice_key_moments.json）+ 完整文本（voice_full_text.json."""
import logging
from typing import List, Dict

from core.path_manager import PathManager

logger = logging.getLogger("module.voice.storage")

from core.base_storage import BaseStorage


class VoiceStorage(BaseStorage):
    """语音结果存储器，负责解耦数据存储动作."""

    def __init__(self, paths: PathManager):
        """初始化."""
        super().__init__(paths, "voice")

    def save_results(self, run_id: str, events: List[Dict]) -> None:
        """保存results."""
        if not events:
            logger.info("没有事件可保存，跳过保存")
            return

        key_moment_events = []
        for event in events:
            key_moment = event.get("key_moment")
            if key_moment:
                key_moment_events.append({
                    "localSec": event.get("localSec"),
                    "key_moment": key_moment,
                    "source": "voice",
                })

        full_text = " ".join(event.get("text", "") for event in events)

        self.save_key_moments(run_id, key_moment_events)
        self.save_full_text(run_id, full_text)

    def save_key_moments(self, run_id: str, events: List[Dict]) -> None:
        """保存keymoments."""
        if not events:
            logger.info("没有关键时刻事件，跳过保存")
            return
        self._save_json_atomic("voice_key_moments.json", run_id, events)

    def save_full_text(self, run_id: str, full_text: str) -> None:
        """保存fulltext."""
        self._save_json_atomic("voice_full_text.json", run_id, {
            "full_text": full_text,
            "word_count": len(full_text),
        })
