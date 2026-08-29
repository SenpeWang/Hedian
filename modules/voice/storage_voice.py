"""语音结果存储：关键事件（voice_key_moments.json）与完整文本（voice_full_text.json）.

从转录产出的事件列表中筛选出带 key_moment 的条目落盘为关键事件，
并把全部事件文本拼接后落盘为完整转写文本。
"""
import logging
from typing import Any, Dict, List

from core.base_storage import BaseStorage
from core.path_manager import PathConfig

logger = logging.getLogger("module.voice.storage")


class VoiceStorage(BaseStorage):
    """语音结果存储器，负责解耦数据存储动作."""

    def __init__(self, paths: PathConfig) -> None:
        """初始化语音存储.

        Args:
            paths: 路径管理器，提供结果目录解析能力.
        """
        super().__init__(paths, "voice")

    def save_results(self, run_id: str, events: List[Dict[str, Any]]) -> None:
        """把事件列表拆分为关键事件与完整文本后分别落盘.

        Args:
            run_id: 本次运行标识.
            events: 转录产出的事件列表，每项含 localSec、key_moment、text.
        """
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

    def save_key_moments(self, run_id: str, events: List[Dict[str, Any]]) -> None:
        """把关键时刻事件原子写入 voice_key_moments.json.

        Args:
            run_id: 本次运行标识.
            events: 时刻事件列表.
        """
        if not events:
            logger.info("没有关键时刻事件，跳过保存")
            return
        self._save_json_atomic("voice_key_moments.json", run_id, events)

    def save_full_text(self, run_id: str, full_text: str) -> None:
        """把完整转写文本与字数原子写入 voice_full_text.json.

        Args:
            run_id: 本次运行标识.
            full_text: 拼接后的完整转写文本.
        """
        self._save_json_atomic("voice_full_text.json", run_id, {
            "full_text": full_text,
            "word_count": len(full_text),
        })
