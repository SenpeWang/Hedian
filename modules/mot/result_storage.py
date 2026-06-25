"""
MOT 结果存储模块

负责保存 MOT 模块的处理结果。
"""
import json
import logging
from pathlib import Path
from typing import List, Dict

from core.path_manager import PathConfig

logger = logging.getLogger("module.mot.storage")


class MOTResultStorage:
    """
    MOT 结果存储器

    负责保存 MOT 模块的处理结果，包括：
    - 关键事件（Mot_key_moments.json）
    - 角色信息（Mot_role_info.json）
    - 关键帧（Mot_key_frames/）
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
            module="mot",
            filename="Mot_key_moments.json",
        )

        try:
            # 确保所有事件都有 roles_details
            for ev in events:
                if "roles_details" not in ev:
                    ev["roles_details"] = {}

            # 原子写入
            tmp_path = str(output_path) + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(events, f, ensure_ascii=False, indent=2)

            import os
            os.replace(tmp_path, str(output_path))

            logger.info(f"保存 {len(events)} 个关键事件到 {output_path}")

        except Exception as e:
            logger.error(f"保存关键事件失败: {e}", exc_info=True)

    def save_role_info(self, run_id: str, roles_info: Dict) -> None:
        """
        保存角色信息

        Args:
            run_id: 运行 ID
            roles_info: 角色信息
        """
        output_path = self._paths.get_result_path(
            run_id=run_id,
            module="mot",
            filename="Mot_role_info.json",
        )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "roles": roles_info,
                    "total_frames": 0,  # 由调用者填充
                    "fps": 0,  # 由调用者填充
                    "video_path": "",  # 由调用者填充
                }, f, ensure_ascii=False, indent=2)

            logger.info(f"保存角色信息到 {output_path}")

        except Exception as e:
            logger.error(f"保存角色信息失败: {e}", exc_info=True)

    def save_panel_violations(self, run_id: str, records: List[Dict]) -> None:
        """
        保存盘台违规记录

        Args:
            run_id: 运行 ID
            records: 违规记录列表
        """
        if not records:
            return

        output_path = self._paths.get_result_path(
            run_id=run_id,
            module="mot",
            filename="Mot_panel_violations.json",
        )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            logger.info(f"保存 {len(records)} 条盘台违规记录到 {output_path}")
        except Exception as e:
            logger.error(f"保存盘台违规记录失败: {e}", exc_info=True)

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
            module="mot",
            filename="Mot_key_moments.json",
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

    def load_role_info(self, run_id: str) -> Dict:
        """
        加载角色信息

        Args:
            run_id: 运行 ID

        Returns:
            角色信息
        """
        input_path = self._paths.get_result_path(
            run_id=run_id,
            module="mot",
            filename="Mot_role_info.json",
        )

        try:
            if not input_path.exists():
                return {}

            with open(input_path, encoding="utf-8") as f:
                role_info = json.load(f)

            logger.info(f"加载角色信息从 {input_path}")
            return role_info

        except Exception as e:
            logger.error(f"加载角色信息失败: {e}", exc_info=True)
            return {}
