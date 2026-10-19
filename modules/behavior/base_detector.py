"""行为检测器抽象基类.

封装行为检测公共能力：冷却期帧数控制、合并式去重与标准化事件结构封装。
"""

from typing import Any, Dict, Optional


class BaseDetector:
    """行为检测器基类，提供冷却 / 合并去重与事件标准化支持."""

    def __init__(self, cooldown_sec: float, fps: float) -> None:
        """初始化行为检测器基类.

        Args:
            cooldown_sec: 事件触发后的冷却时间（秒）.
            fps: 视频流的采样帧率.
        """
        self.cooldown_sec: float = cooldown_sec
        self.fps: float = fps
        self.cooldown_frames: int = int(fps * cooldown_sec)
        self._last_event_frame: int = -10_000
        # 合并式去重状态：最近一次已记录事件的帧号与角色
        self._last_merged_frame: int = -10_000
        self._last_merged_role: Optional[str] = None

    def _cooldown_ok(self, frame_count: int) -> bool:
        """检查当前帧是否已经脱离冷却期.

        Args:
            frame_count: 当前视频帧号.

        Returns:
            脱离冷却期返回 True，仍在冷却期返回 False.
        """
        return frame_count - self._last_event_frame >= self.cooldown_frames

    def _mark_event(self, frame_count: int) -> None:
        """记录事件触发时的帧号，启动冷却计时.

        Args:
            frame_count: 当前视频帧号.
        """
        self._last_event_frame = frame_count

    def _merge_ready(
        self,
        role: Optional[str],
        frame_count: int,
        merge_sec: float,
        role_cooldown_sec: float,
    ) -> bool:
        """合并式去重判定：持续指向只记一次.

        与最近一次已记录事件比较——
          - 同角色且间隔 <= merge_sec：视为同一次，合并；
          - 角色不同但间隔 <= role_cooldown_sec：视为角色抖动，合并；
          - 其余情况记录新事件，并推进合并基准。

        角色为 None 表示本帧姿态模型未能判定（漏检），不视为换了人：
        沿用上一次已记录的角色参与合并，避免同一动作被切成多段。

        Args:
            role: 当前判定的行为人角色，不可判定时为 None.
            frame_count: 当前视频帧号.
            merge_sec: 同角色事件的合并窗口（秒）.
            role_cooldown_sec: 角色抖动冷却（秒）.

        Returns:
            需要记录新事件时返回 True.
        """
        if self._last_merged_frame >= 0:
            effective_role = (
                role if role is not None else self._last_merged_role
            )
            frame_gap = frame_count - self._last_merged_frame
            same_role = effective_role == self._last_merged_role
            if same_role and frame_gap <= int(merge_sec * self.fps):
                return False
            if (not same_role) and frame_gap <= int(
                role_cooldown_sec * self.fps
            ):
                return False
        self._last_merged_frame = frame_count
        self._last_merged_role = role or self._last_merged_role
        return True

    def _make_event(
        self,
        event_type: str,
        state: str,
        frame_count: int,
        extra: Dict[str, Any],
        role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建标准化的行为事件字典.

        Args:
            event_type: 行为事件标识类型.
            state: 行为状态描述文本.
            frame_count: 事件发生时的视频帧号.
            extra: 额外附加字段.
            role: 行为人角色，不可判定时为 None.

        Returns:
            标准格式的行为事件字典.
        """
        return {
            "event": event_type,
            "state": state,
            "frame_id": frame_count,
            "role": role,
            **extra,
        }
