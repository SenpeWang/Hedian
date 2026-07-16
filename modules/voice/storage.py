"""
语音结果存储模块
"""
import json
import logging
import os
from pathlib import Path
from typing import List, Dict

from core.path_manager import PathConfig

logger = logging.getLogger("module.voice.storage")

class VoiceResultStorage:
    """语音结果存储器，负责解耦数据存储动作"""

    def __init__(self, paths: PathConfig):
        self._paths = paths

    def save_key_moments(self, run_id: str, events: List[Dict]) -> None:
        """保存关键时刻 (Voice_key_moments.json)"""
        output_path = self._paths.get_result_path(
            run_id=run_id,
            module="voice",
            filename="Voice_key_moments.json",
        )
        try:
            tmp_path = str(output_path) + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(events, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, str(output_path))
            logger.info(f"保存 {len(events)} 个关键时刻到 {output_path}")
        except Exception as e:
            logger.error(f"保存关键时刻失败: {e}", exc_info=True)

    def save_full_text(self, run_id: str, full_text: str) -> None:
        """保存完整文本 (Voice_full_text.json)"""
        output_path = self._paths.get_result_path(
            run_id=run_id,
            module="voice",
            filename="Voice_full_text.json",
        )
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "full_text": full_text,
                    "word_count": len(full_text),
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"保存完整文本到 {output_path}")
        except Exception as e:
            logger.error(f"保存完整文本失败: {e}", exc_info=True)
