"""行为检测器抽象基类.

封装行为检测公共能力：冷却期帧数控制与标准化事件结构封装。
"""
from typing import Any, Dict


class BaseDetector:
    """行为检测器基类：提供冷却去重与事件标准化支持."""

    def __init__(self, cooldown_sec: float, fps: float):
        """初始化行为检测器基类.

        Args:
            cooldown_sec (float): 事件触发后的冷却时间（秒）.
            fps (float): 视频流的采样帧率.
        """
        self.cooldown_frames: int = int(fps * cooldown_sec)
        self._last_event_frame: int = -10_000

    def _cooldown_ok(self, frame_count: int) -> bool:
        """检查当前帧是否已经脱离冷却期.

        Args:
            frame_count (int): 当前视频帧号.

        Returns:
            bool: 脱离冷却期返回 True，仍在冷却期返回 False.
        """
        return frame_count - self._last_event_frame >= self.cooldown_frames

    def _mark_event(self, frame_count: int) -> None:
        """记录事件触发时的帧号，启动冷却计时.

        Args:
            frame_count (int): 当前视频帧号.
        """
        self._last_event_frame = frame_count

    def _make_event(
        self,
        event_type: str,
        state: str,
        frame_count: int,
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构建标准化的行为事件字典.

        Args:
            event_type (str): 行为事件标识类型.
            state (str): 行为状态描述文本.
            frame_count (int): 事件发生时的视频帧号.
            extra (Dict[str, Any]): 额外附加字段.

        Returns:
            Dict[str, Any]: 标准格式的行为事件字典.
        """
        return {
            "event": event_type,
            "state": state,
            "frame_id": frame_count,
            "operator": None,
            **extra,
        }
