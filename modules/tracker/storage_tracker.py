"""Tracker 结果存储：统一管理与原子落盘多目标跟踪的关键事件、角色分配与关键帧."""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core.base_storage import BaseStorage
from core.path_manager import PathManager

logger = logging.getLogger("module.tracker.storage")

KEY_MOMENTS_FILENAME = "tracker_key_moments.json"
ROLE_INFO_FILENAME = "tracker_role_info.json"


class TrackerStorage(BaseStorage):
    """多目标跟踪结果存储管理器.

    封装跟踪模块所有产出数据（关键时刻事件、工位角色分配信息、可视化关键帧截图）
    的结构化管理与原子落盘能力。
    """

    def __init__(self, paths: PathManager):
        """初始化跟踪结果存储器.

        Args:
            paths (PathManager): 路径管理器实例.
        """
        super().__init__(paths, "tracker")

    def save_key_moments(
        self, run_id: str, events: List[Dict[str, Any]]
    ) -> None:
        """全量原子保存跟踪关键时刻事件列表.

        Args:
            run_id (str): 本次运行标识.
            events (List[Dict[str, Any]]): 待保存的事件列表.
        """
        if not events:
            logger.info("没有跟踪事件可保存，跳过落盘")
            return
        sanitized_events = []
        for event_item in events:
            sanitized_item = dict(event_item)
            sanitized_item.setdefault("source", "tracker")
            sanitized_events.append(sanitized_item)
        self._save_json_atomic(KEY_MOMENTS_FILENAME, run_id, sanitized_events)

    def save_role_info(self, run_id: str, identity_map: Dict[str, int]) -> None:
        """原子保存角色与工位分配映射信息.

        Args:
            run_id (str): 本次运行标识.
            identity_map (Dict[str, int]): 身份角色到 track_id 的映射，例如 {"LEADER": 1, "ROAD1": 2}.
        """
        valid_identity_map = {
            identity_name: int(track_id)
            for identity_name, track_id in identity_map.items()
            if identity_name in ("LEADER", "ROAD1", "ROAD2")
        }
        self._save_json_atomic(
            ROLE_INFO_FILENAME, run_id, {"roles": valid_identity_map}
        )

    def get_key_frames_dir(self, run_id: str) -> Path:
        """获取并自动创建关键帧存储目录.

        Args:
            run_id (str): 本次运行标识.

        Returns:
            Path: 关键帧所在目录的 Path 对象.
        """
        key_frames_dir = self._paths.get_result_dir(run_id, "tracker") / "key_frames"
        key_frames_dir.mkdir(parents=True, exist_ok=True)
        return key_frames_dir

    def save_key_frame(
        self,
        run_id: str,
        filename: str,
        frame: np.ndarray,
    ) -> Optional[Path]:
        """原子保存一张关键帧截图.

        Args:
            run_id (str): 本次运行标识.
            filename (str): 文件名称（如 role_assigned_12.5s.jpg）.
            frame (np.ndarray): 图像矩阵（BGR 格式）.

        Returns:
            Optional[Path]: 成功保存后的文件绝对路径；失败返回 None.
        """
        try:
            key_frames_dir = self.get_key_frames_dir(run_id)
            target_path = key_frames_dir / filename
            temp_path = key_frames_dir / f"{filename}.tmp.jpg"
            encode_success = cv2.imwrite(str(temp_path), frame)
            if encode_success:
                os.replace(temp_path, target_path)
                logger.info(f"保存关键帧: {target_path}")
                return target_path
            else:
                logger.error(f"cv2.imwrite 编码写入关键帧失败: {target_path}")
                return None
        except Exception as write_error:
            logger.error(f"保存关键帧异常 {filename}: {write_error}", exc_info=True)
            return None
