"""身份监控模块结果存储.

输出目录与命名遵循框架契约：`{result_root}/{run_id}/identity/*.json`
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.base_storage import BaseStorage

logger = logging.getLogger("module.identity.storage")

KEY_MOMENTS_FILENAME = "identity_key_moments.json"
AWAY_FILENAME = "identity_away.json"
ROLE_INFO_FILENAME = "identity_role_info.json"


class IdentityStorage(BaseStorage):
    """身份监控结果存储（key_moments / 离岗明细 / 角色快照）."""

    def __init__(self, paths) -> None:
        """初始化.

        Args:
            paths: 路径配置, 用于解析结果输出目录.
        """
        super().__init__(paths, "identity")

    def save_key_moments(self, run_id: str, events: List[Dict[str, Any]]) -> bool:
        """保存关键事件列表（前端时间轴与评估层读取的主产物）.

        Args:
            run_id: 本次运行标识.
            events: 事件列表，元素至少含 localSec / key_moment / source.

        Returns:
            保存成功返回 True.
        """
        payload = {
            "source": "identity",
            "count": len(events),
            "events": events,
        }
        return self._save_json_atomic(KEY_MOMENTS_FILENAME, run_id, payload)

    def save_away_events(self, run_id: str, events: List[Dict[str, Any]]) -> bool:
        """保存离岗明细（便于人工回溯，非评估层必需）.

        Args:
            run_id: 本次运行标识.
            events: 离岗段列表.

        Returns:
            保存成功返回 True.
        """
        payload = {
            "source": "identity",
            "count": len(events),
            "away_events": events,
        }
        return self._save_json_atomic(AWAY_FILENAME, run_id, payload)

    def save_role_info(
        self,
        run_id: str,
        roles: Dict[str, Any],
        room_id: str,
    ) -> bool:
        """保存最后一次身份/工位快照.

        Args:
            run_id: 本次运行标识.
            roles: 身份 -> 快照信息（tid / 在岗 / 离岗秒数等）.
            room_id: 机室编号.

        Returns:
            保存成功返回 True.
        """
        payload = {"room_id": str(room_id), "roles": roles}
        return self._save_json_atomic(ROLE_INFO_FILENAME, run_id, payload)
