"""
举手检测模块

使用 YOLOPose 检测举手动作，带投票确认和冷却期。
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger("module.behavior.hand_raiser")


class HandRaiser:
    """
    举手检测器

    检测流程：
    1. YOLOPose 检测姿态关键点
    2. 判断手腕是否高于肩膀
    3. 投票确认（vote_window 帧中 vote_threshold 帧举起）
    4. 冷却期（cooldown_frames 帧内不重复触发）
    """

    def __init__(
        self,
        detector,
        vote_window: int = 3,
        vote_threshold: int = 2,
        cooldown_frames: int = 300,
    ):
        """
        初始化举手检测器

        Args:
            detector: 目标检测器
            vote_window: 投票窗口大小
            vote_threshold: 投票阈值
            cooldown_frames: 冷却帧数
        """
        self._detector = detector
        self._vote_window = vote_window
        self._vote_threshold = vote_threshold
        self._cooldown_frames = cooldown_frames
        self._buffers = {}  # {role: deque}
        self._last_raise_frame = {}  # {role: frame_count}

    def check(
        self,
        frame: np.ndarray,
        tracks: list,
        roles_assigned: bool,
        frame_count: int,
        roles: dict = None,
    ) -> Optional[str]:
        """
        检查是否有举手

        Args:
            frame: 视频帧
            tracks: 跟踪轨迹列表
            roles_assigned: 是否已分配角色
            frame_count: 当前帧号
            roles: 角色映射 {"LEADER": track, "ROAD1": track, ...}

        Returns:
            举手的角色名，或 None
        """
        if not roles_assigned or frame_count % 5 != 0:
            return None

        poses = self._detector.detect_pose(frame)

        for role_name in ("ROAD1", "ROAD2"):
            # 冷却期检查
            in_cooldown = (
                frame_count - self._last_raise_frame.get(role_name, -9999)
                < self._cooldown_frames
            )
            if in_cooldown:
                continue

            # 获取对应角色的跟踪
            rt = None
            if roles:
                rt = roles.get(role_name)
            if rt is None:
                continue

            rc = rt.get_center()

            # 找最近的姿态
            best_pose, best_d = None, float("inf")
            for p in poses:
                d = np.linalg.norm(rc - p["center"])
                if d < best_d:
                    best_d, best_pose = d, p

            # 判断举手
            raised = (
                best_pose is not None
                and self._detector.check_hand_raised(best_pose["keypoints"])
            )

            # 投票缓冲
            if role_name not in self._buffers:
                self._buffers[role_name] = deque(maxlen=self._vote_window)
            self._buffers[role_name].append(raised)

            buf = self._buffers[role_name]
            if len(buf) == self._vote_window and sum(buf) >= self._vote_threshold:
                self._last_raise_frame[role_name] = frame_count
                self._buffers[role_name].clear()
                return role_name

        return None

    def reset(self) -> None:
        """重置状态"""
        self._buffers.clear()
        self._last_raise_frame.clear()
