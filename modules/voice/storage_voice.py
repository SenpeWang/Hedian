"""
语音结果存储模块

负责保存 Voice 模块的处理结果，包括：
- 关键事件（voice_key_moments.json）
- 完整文本（voice_full_text.json）
"""
import json
import logging
import os
from typing import List, Dict

from core.path_manager import PathConfig

logger = logging.getLogger("module.voice.storage")


from core.base_storage import BaseStorage

class VoiceResultStorage(BaseStorage):
    """语音结果存储器，负责解耦数据存储动作"""

    def __init__(self, paths: PathConfig):
        super().__init__(paths, "voice")

    def save_results(self, run_id: str, events: List[Dict]) -> None:
        """
        保存语音结果（高层入口）：key_moments + full_text

        内部完成：
        1. 空事件检查
        2. 提取 key_moment 事件（仅 localSec + key_moment + source）
        3. 拼接完整文本
        4. 调用 save_key_moments + save_full_text
        """
        if not events:
            logger.info("没有事件可保存，跳过保存")
            return

        # 提取 key_moment 事件
        key_moment_events = []
        for event in events:
            key_moment = event.get("key_moment")
            if key_moment:
                key_moment_events.append({
                    "localSec": event.get("localSec"),
                    "key_moment": key_moment,
                    "source": "voice",
                })

        # 拼接完整文本
        full_text = " ".join(event.get("text", "") for event in events)

        self.save_key_moments(run_id, key_moment_events)
        self.save_full_text(run_id, full_text)

    def save_key_moments(self, run_id: str, events: List[Dict]) -> None:
        """
        保存关键时刻 (voice_key_moments.json)

        空事件列表会跳过保存。
        """
        if not events:
            logger.info("没有关键时刻事件，跳过保存")
            return
        self._save_json_atomic("voice_key_moments.json", run_id, events)

    def save_full_text(self, run_id: str, full_text: str) -> None:
        """保存完整文本 (voice_full_text.json)"""
        self._save_json_atomic("voice_full_text.json", run_id, {
            "full_text": full_text,
            "word_count": len(full_text),
        })
