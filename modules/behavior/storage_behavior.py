"""Behavior 结果存储：统一管理与落盘所有行为关键事件.

本模块以面向对象方式统管行为类事件（举手、手指屏幕、手指文件等），
所有事件通过统一入口追加落盘到 behavior_key_moments.json。
"""
import json
import logging
import os
from typing import Dict, List, Optional, Set

from core.base_storage import BaseStorage
from core.event_bus import EventStream, EventTopic
from core.path_manager import PathManager

logger = logging.getLogger("module.behavior.storage")

FILENAME = "behavior_key_moments.json"
VALID_IDENTITIES: Set[str] = {"LEADER", "ROAD1", "ROAD2"}


class BehaviorStorage(BaseStorage):
    """行为事件存储：以面向对象方式统一落盘所有行为关键事件。

    举手与手指类事件均通过统一方法（add_event / report_hand_raise）
    原子追加到 behavior_key_moments.json，避免多进程写入互相覆盖。
    """

    def __init__(self, paths: PathManager):
        """初始化行为存储.

        Args:
            paths (PathManager): 路径管理器，提供结果目录解析能力.
        """
        super().__init__(paths, "behavior")

    def _read_events(self, run_id: str) -> List[Dict]:
        """读取已有行为事件列表.

        Args:
            run_id (str): 本次运行标识.

        Returns:
            List[Dict]: 已有事件列表；文件不存在或损坏时返回空列表.
        """
        file_path = self._paths.get_result_path(run_id, "behavior", FILENAME)
        if not os.path.exists(file_path):
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as file_obj:
                data = json.load(file_obj)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as read_error:
            logger.warning(f"读取 {file_path} 失败: {read_error}")
            return []

    def add_event(self, run_id: str, event: Dict) -> None:
        """原子追加单条行为事件.

        举手与手指类事件统一通过本方法落盘，保证多进程追加安全。

        Args:
            run_id (str): 本次运行标识.
            event (Dict): 行为事件字典，约定包含 localSec 与 key_moment 字段.
        """
        events_list = self._read_events(run_id)
        events_list.append(event)
        self._save_json_atomic(FILENAME, run_id, events_list)

    def report_hand_raise(
        self,
        event_bus: EventStream,
        run_id: str,
        identity: Optional[str],
        timestamp: float,
        track_id: Optional[int] = None,
        inference_fn=None,
    ) -> None:
        """按行为类规则上报一次举手事件.

        完成三件事：
        1. 落盘到 behavior_key_moments.json（与手指类同一文件）
           - 已分配身份：记录为 '{identity}举手'（如 'ROAD1举手'）
           - 未分配身份：记录为 '举手'
        2. 推送推理流供前端展示（署名 behavior）
        3. 推送事件流供 web 规则层判定（EventTopic.BEHAVIOR_HAND_RAISED）

        Args:
            event_bus (EventStream): 事件总线，用于推送推理流与事件流.
            run_id (str): 本次运行标识.
            identity (Optional[str]): 举手者的合法身份 ('LEADER'|'ROAD1'|'ROAD2')；未分配严格为 None.
            timestamp (float): 事件发生时间（秒）.
            track_id (Optional[int]): 跟踪目标全局唯一 ID，可选.
        """
        local_sec = round(timestamp, 2)
        valid_identity = identity if identity in VALID_IDENTITIES else None
        key_moment = f"{valid_identity}举手" if valid_identity else "举手"

        self.add_event(run_id, {
            "localSec": local_sec,
            "key_moment": key_moment,
            "source": "behavior",
        })

        display_data = {
            "state": "举手",
            "operator": valid_identity,
        }
        if track_id is not None:
            display_data["track_id"] = track_id

        if inference_fn is not None:
            inference_fn("behavior", {
                "localSec": local_sec,
                "tag": "HAND_RAISED",
                "data": display_data,
            })
        elif hasattr(event_bus, "push_display"):
            event_bus.push_display("behavior", {
                "localSec": local_sec,
                "tag": "HAND_RAISED",
                "data": display_data,
            })

        event_payload = {
            "localSec": local_sec,
            "operator": valid_identity,
        }
        if track_id is not None:
            event_payload["track_id"] = track_id

        event_bus.publish(
            EventTopic.BEHAVIOR_HAND_RAISED,
            event_payload,
            ts=timestamp,
        )

        logger.info(
            f"举手上报: {key_moment}"
            f" {'(track_id=' + str(track_id) + ')' if track_id is not None else ''}"
            f" @{local_sec}s [source=behavior]"
        )
