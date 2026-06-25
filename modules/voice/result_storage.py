"""
语音结果存储模块

负责保存语音模块的处理结果。
"""
import json
import logging
from pathlib import Path
from typing import List, Dict

from core.path_manager import PathConfig

logger = logging.getLogger("module.voice.storage")


class VoiceResultStorage:
    """
    语音结果存储器

    负责保存语音模块的处理结果，包括：
    - 关键事件（Voice_key_moments.json）
    - 完整文本（Voice_full_text.json）
    """

    def __init__(self, paths: PathConfig):
        """
        初始化存储器

        Args:
            paths: 路径配置
        """
        self._paths = paths

    def save_key_moments(self, run_id: str, events: List[Dict]) -> None:
        """
        保存关键事件

        Args:
            run_id: 运行 ID
            events: 事件列表
        """
        output_path = self._paths.get_result_path(
            run_id=run_id,
            module="voice",
            filename="Voice_key_moments.json",
        )

        try:
            # 原子写入
            tmp_path = str(output_path) + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(events, f, ensure_ascii=False, indent=2)

            import os
            os.replace(tmp_path, str(output_path))

            logger.info(f"保存 {len(events)} 个关键事件到 {output_path}")

        except Exception as e:
            logger.error(f"保存关键事件失败: {e}", exc_info=True)

    def save_full_text(self, run_id: str, full_text: str) -> None:
        """
        保存完整文本

        Args:
            run_id: 运行 ID
            full_text: 完整文本
        """
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

    def load_key_moments(self, run_id: str) -> List[Dict]:
        """
        加载关键事件

        Args:
            run_id: 运行 ID

        Returns:
            事件列表
        """
        input_path = self._paths.get_result_path(
            run_id=run_id,
            module="voice",
            filename="Voice_key_moments.json",
        )

        try:
            if not input_path.exists():
                return []

            with open(input_path, encoding="utf-8") as f:
                events = json.load(f)

            logger.info(f"加载 {len(events)} 个关键事件从 {input_path}")
            return events

        except Exception as e:
            logger.error(f"加载关键事件失败: {e}", exc_info=True)
            return []
