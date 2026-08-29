"""
信息通报制度.

实现 BaseRule 接口，管理信息通报流程。

步骤：
(1)信息发起者需要举手高声喊出“信息通报”或“信息通告”；
(2)在主控室的其他成员听到“信息通报”或“信息通告”后，立刻停下手中正在进行的工作接受信息；
(3)信息发起者在确认团队成员均予以关注，进行后续的信息传递；
(4)信息传递结束，由信息发起者喊出“通报完毕”，确认信息结束；
(5)收到“收到”等语音给予回应。
"""
import logging
from typing import Any, Dict, Optional

from core.event_bus import EventBus, EventTopic
from rules.rule_base import BaseRule

logger = logging.getLogger("rules.info_notice")


class InfoNoticeRule(BaseRule):
    """信息通报制度."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化信息通报制度.

        Args:
            config: 制度配置字典，可为 None.
        """
        self._config = config or {}
        self._event_bus: Optional[EventBus] = None

        self._active = False
        self._flow_id = 0
        self._flow_counter = 0
        self._flow_start_sec = 0

        # 时间记录，用于判定 5 秒内 举手 + 声音 进入流程
        self._last_hand_raise_ts = -999.0

        # 内容检查清单
        self._checklist = {
            "raise_hand_and_shout": False,
            "others_stopped_and_listened": False,
            "others_attended": False,
            "shout_finished": False,
            "received_acknowledged": False,
        }

    def name(self) -> str:
        """制度名称."""
        return "info_notice"

    def subscribe_events(self, event_bus: EventBus) -> None:
        """订阅本制度关心的事件.

        Args:
            event_bus: 事件总线.
        """
        self._event_bus = event_bus
        event_bus.subscribe(EventTopic.VOICE_KEY_MOMENT, self._on_voice_intent)
        event_bus.subscribe(EventTopic.BEHAVIOR_HAND_RAISED,
                            self._on_hand_raised)
        event_bus.subscribe(EventTopic.GAZE_ATTENTION, self._on_gaze_status)

    def _start_flow(self, ts: float, source: str) -> None:
        """启动信息通报流程.

        Args:
            ts: 流程开始时间（秒）.
            source: 流程触发来源（如 voice）.
        """
        self._active = True
        self._flow_id = self._next_flow_id()
        self._flow_start_sec = ts

        # 判定是否伴随举手：如果在发出语音前的 5 秒内有举手动作，则判定为“举手+通报”
        has_hand_raise = (0 <= (ts - self._last_hand_raise_ts) <= 5.0)

        self._checklist = {
            "raise_hand_and_shout": has_hand_raise,
            "others_stopped_and_listened": False,
            "others_attended": False,
            "shout_finished": False,
            "received_acknowledged": False,
        }

        if self._event_bus:
            self._event_bus.publish(EventTopic.FLOW_STARTED, {
                "flow_id": self._flow_id,
                "flow_type": "info_notice",
                "flow_start_sec": ts,
                "start_source": source,
            },
                                    ts=ts)

        logger.info(
            f"信息通报流程开始 flow_id={self._flow_id} @{ts:.1f}s 伴随举手={has_hand_raise} source={source}"
        )

    def _close_flow(
        self, ts: float = 0, source: str = "unknown"
    ) -> Dict[str, Any]:
        """关闭信息通报流程并发布 FLOW_ENDED 事件.

        Args:
            ts: 流程结束时间（秒）.
            source: 结束来源（如 normal_end）.

        Returns:
            流程事件字典；无活跃流程时返回 None.
        """
        if not self._active:
            return None

        flow = {
            "flow_id": self._flow_id,
            "flow_type": "info_notice",
            "flow_start_sec": self._flow_start_sec,
            "flow_end_sec": ts,
            "flow_continue_sec": round(ts - self._flow_start_sec, 2),
            "end_source": source,
            "content_checklist": dict(self._checklist),
        }

        if self._event_bus:
            self._event_bus.publish(EventTopic.FLOW_ENDED, flow, timestamp=ts)

        logger.info(
            f"信息通报流程结束 flow_id={self._flow_id} @{ts:.1f}s source={source}")

        self._active = False
        self._flow_id = 0
        self._last_hand_raise_ts = -999.0

        return flow

    def _on_hand_raised(self, event: Dict[str, Any]) -> None:
        """处理 Behavior 举手（BEHAVIOR_HAND_RAISED 事件流，payload={localSec, operator}）."""
        payload = event.get("data", {})
        ts = payload.get("localSec", event.get("ts", 0))

        # 记录举手时间，用于 _start_flow 判定 5 秒内是否伴随举手
        self._last_hand_raise_ts = ts
        logger.debug(f"信息通报: 收到举手事件 @{ts:.1f}s（单独举手不触发信息通报流程）")

    def _on_voice_intent(self, event: Dict[str, Any]) -> None:
        """处理语音事件（事件流仅包含 localSec 和 key_moment 字段）."""
        payload = event.get("data", {})
        key_moment = payload.get("key_moment", "")
        ts = payload.get("localSec", event.get("ts", 0.0))
        if not key_moment:
            return

        # 超时自动关闭
        if self._active and ts - self._flow_start_sec > 180.0:
            logger.warning(f"信息通报流程超时未结束，自动关闭 flow_id={self._flow_id}")
            self._close_flow(ts, source="timeout")
            return

        if key_moment in ["信息通报", "信息通告"]:
            logger.debug(f"信息通报: 收到信息通报语音事件 @{ts:.1f}s，开始流程")
            self._start_flow(ts, source="voice")
        elif key_moment in ["通报完毕", "通告完毕"]:
            if self._active:
                self._checklist["shout_finished"] = True
                logger.info(f"信息通报: 喊出'通报完毕' @{ts:.1f}s，流程结束")
                self._close_flow(ts, source="normal_end")
        elif key_moment == "收到":
            if self._active:
                self._checklist["received_acknowledged"] = True
                logger.info(f"信息通报: 收到'收到'语音回应 @{ts:.1f}s")

    def _on_gaze_status(self, event: Dict[str, Any]) -> None:
        """处理 Gaze 关注度状态（GAZE_ATTENTION 事件流，payload={localSec, has_turned, displacement, ...}）."""
        if not self._active:
            return
        payload = event.get("data", {})
        ts = payload.get("localSec", event.get("ts", 0))
        has_turned = payload.get("has_turned", False)

        if has_turned:
            self._checklist["others_attended"] = True
            self._checklist["others_stopped_and_listened"] = True
            logger.info(f"信息通报: 团队成员均予以关注 @{ts:.1f}s")

    def reset(self) -> None:
        """重置信息通报状态，防止新一轮推理混入旧的举手/关注状态."""
        super().reset()
        self._checklist = {
            "raise_hand_and_shout": False,
            "others_stopped_and_listened": False,
            "others_attended": False,
            "shout_finished": False,
            "received_acknowledged": False,
        }
        self._last_hand_raise_ts = -999.0


def register() -> InfoNoticeRule:
    """实例化并返回信息通报制度实例.

    Returns:
        可注册到注册表的信息通报制度实例.
    """
    return InfoNoticeRule()
