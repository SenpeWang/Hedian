"""
信息通报制度

实现 BaseRule 接口，管理信息通报流程。

步骤：
(1)信息发起者需要举手高声喊出“信息通报”或“信息通告”；
(2)在主控室的其他成员听到“信息通报”或“信息通告”后，立刻停下手中正在进行的工作接受信息；
(3)信息发起者在确认团队成员均予以关注，进行后续的信息传递；
(4)信息传递结束，由信息发起者喊出“通报完毕”，确认信息结束；
(5)收到“收到”等语音给予回应。
"""
import logging

from core.event_bus import EventBus, EventTopic
from rules.rule_base import BaseRule

logger = logging.getLogger("rules.info_notice")


class InfoNoticeRule(BaseRule):
    """信息通报制度"""

    def __init__(self, config: dict = None):
        """
        初始化信息通报制度

        Args:
            config: 配置字典
        """
        self._config = config or {}
        self._event_bus = None

        self._active = False
        self._flow_id = 0
        self._flow_counter = 0
        self._flow_start_sec = 0

        # 时间记录，用于判定 5 秒内 举手 + 声音 进入流程
        self._last_hand_raise_ts = -999.0
        self._last_voice_shout_ts = -999.0

        # 内容检查清单
        self._checklist = {
            "raise_hand_and_shout": False,
            "others_stopped_and_listened": False,
            "others_attended": False,
            "shout_finished": False,
            "received_acknowledged": False,
        }

    def name(self) -> str:
        """制度名称"""
        return "info_notice"

    def subscribe_events(self, event_bus: EventBus) -> None:
        """订阅事件"""
        self._event_bus = event_bus
        event_bus.subscribe(EventTopic.VOICE_KEY_MOMENT, self._on_voice_intent)
        event_bus.subscribe(EventTopic.TRACKER_HAND_RAISED, self._on_hand_raised)
        event_bus.subscribe(EventTopic.GAZE_ATTENTION, self._on_gaze_status)
        event_bus.subscribe(EventTopic.FLOW_ENDED, self._on_flow_ended)

    def is_active(self) -> bool:
        """是否有活跃流程"""
        return self._active

    def get_current_flow(self) -> dict:
        """获取当前活跃流程"""
        if not self._active:
            return None
        return {
            "flow_id": self._flow_id,
            "flow_type": "info_notice",
            "flow_start_sec": self._flow_start_sec,
            "content_checklist": self._checklist.copy(),
        }

    def finalize(self) -> dict:
        """视频结束时关闭流程"""
        if not self._active:
            return None
        return self._close_flow(source="finalize")

    def _next_flow_id(self) -> int:
        """获取下一个流程 ID"""
        self._flow_counter += 1
        return self._flow_counter

    def _start_flow(self, ts: float, source: str) -> None:
        """启动信息通报流程"""
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
            }, ts=ts)

        logger.info(f"信息通报流程开始 flow_id={self._flow_id} @{ts:.1f}s 伴随举手={has_hand_raise} source={source}")

    def _close_flow(self, ts: float = 0, source: str = "unknown") -> dict:
        """关闭信息通报流程"""
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
            self._event_bus.publish(EventTopic.FLOW_ENDED, flow, ts=ts)

        logger.info(f"信息通报流程结束 flow_id={self._flow_id} @{ts:.1f}s source={source}")

        self._active = False
        self._flow_id = 0
        self._last_hand_raise_ts = -999.0
        self._last_voice_shout_ts = -999.0

        return flow

    def _on_hand_raised(self, msg: dict) -> None:
        """处理 MOT 举手"""
        data = msg.get("data", {})
        event = data.get("event", "")
        ts = data.get("localSec", msg.get("ts", 0))

        if event == "HAND_RAISED":
            self._last_hand_raise_ts = ts
            logger.debug(f"信息通报: 收到举手事件 @{ts:.1f}s（单独举手不触发信息通报流程）")

    def _on_voice_intent(self, msg: dict) -> None:
        """处理语音事件（仅包含 localSec 和 key_moment 字段）"""
        data = msg.get("data", {})
        key_moment = data.get("key_moment", "")
        ts = data.get("localSec", msg.get("ts", 0.0))
        if not key_moment:
            return

        # 超时自动关闭
        if self._active and ts - self._flow_start_sec > 180.0:
            logger.warning(f"信息通报流程超时未结束，自动关闭 flow_id={self._flow_id}")
            self._close_flow(ts, source="timeout")
            return

        if key_moment in ["信息通报", "信息通告"]:
            self._last_voice_shout_ts = ts
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

    def _on_gaze_status(self, msg: dict) -> None:
        """处理 Gaze 关注度状态"""
        if not self._active:
            return
        data = msg.get("data", {})
        event = data.get("event", "")
        ts = data.get("localSec", msg.get("ts", 0))

        if event == "ATTENTION_RESULT":
            has_turned = data.get("has_turned", False)
            if has_turned:
                self._checklist["others_attended"] = True
                self._checklist["others_stopped_and_listened"] = True
                logger.info(f"信息通报: 团队成员均予以关注 @{ts:.1f}s")

    def _on_flow_ended(self, msg: dict) -> None:
        """处理其他流程结束事件（互斥）"""
        if not self._active:
            return
        data = msg.get("data", {})
        flow_id = data.get("flow_id", 0)
        flow_type = data.get("flow_type", "")
        ts = data.get("localSec", msg.get("ts", 0))

        # 如果是本流程的结束事件，不作处理
        if flow_type == "info_notice" and flow_id == self._flow_id:
            return

        # 其他流程强制开启/结束，导致本流程强制关闭
        logger.info(f"信息通报: 检测到其他流程 {flow_type}#{flow_id}，本流程强制关闭")
        self._close_flow(ts, source="conflict")




def register():
    """模块注册入口"""
    return InfoNoticeRule()
