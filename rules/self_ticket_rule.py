"""
自唱票制度

实现 BaseRegulation 接口，管理自唱票流程。

自唱票流程：
  开始：读出九字码（语音检测到设备码重复）
  内容：操作员读出设备九字码 → 执行操作 → 确认
  结束：下一个自唱票开始 或 监护制结束
"""
import logging

from core.event_bus import EventBus, EventTopic
from rules.rule_base import BaseRule

logger = logging.getLogger("rules.self_ticket")

class SelfTicketRule(BaseRule):
    """自唱票制度"""

    def __init__(self, config: dict = None):
        """初始化自唱票制度"""
        self._config = config or {}
        self._event_bus = None

        self._active = False
        self._flow_id = 0
        self._flow_counter = 0
        self._flow_start_sec = 0
        self._device_code = ""

        # 内容检查
        self._code_read = False
        self._operation_executed = False
        self._confirm_closed = False

    def name(self) -> str:
        """制度名称"""
        return "self_ticket"

    def subscribe_events(self, event_bus: EventBus) -> None:
        """订阅事件"""
        self._event_bus = event_bus
        event_bus.subscribe(EventTopic.VOICE_KEY_MOMENT, self._on_voice_intent)

    def is_active(self) -> bool:
        """是否有活跃流程"""
        return self._active

    def get_current_flow(self) -> dict:
        """获取当前活跃流程"""
        if not self._active:
            return None
        return {
            "flow_id": self._flow_id,
            "flow_type": "self_ticket",
            "flow_start_sec": self._flow_start_sec,
            "device_code": self._device_code,
            "code_read": self._code_read,
            "operation_executed": self._operation_executed,
            "confirm_closed": self._confirm_closed,
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

    def _start_flow(self, ts: float, device_code: str) -> None:
        """启动自唱票流程"""
        # 关闭上一个自唱票
        if self._active:
            self._close_flow(ts, source="new_ticket")

        self._active = True
        self._flow_id = self._next_flow_id()
        self._flow_start_sec = ts
        self._device_code = device_code
        self._code_read = False
        self._operation_executed = False
        self._confirm_closed = False

        if self._event_bus:
            self._event_bus.publish(EventTopic.FLOW_STARTED, {
                "flow_id": self._flow_id,
                "flow_type": "self_ticket",
                "flow_start_sec": ts,
                "device_code": device_code,
            }, ts=ts)

        logger.info(f"流程开始 flow_id={self._flow_id} @{ts:.1f}s 设备={device_code}")

    def _close_flow(self, ts: float = 0, source: str = "unknown") -> dict:
        """关闭自唱票流程"""
        if not self._active:
            return None

        flow = {
            "flow_id": self._flow_id,
            "flow_type": "self_ticket",
            "flow_start_sec": self._flow_start_sec,
            "flow_end_sec": ts,
            "flow_continue_sec": round(ts - self._flow_start_sec, 2),
            "end_source": source,
            "device_code": self._device_code,
            "code_read": self._code_read,
            "operation_executed": self._operation_executed,
            "confirm_closed": self._confirm_closed,
        }

        if self._event_bus:
            self._event_bus.publish(EventTopic.FLOW_ENDED, flow, ts=ts)

        logger.info(f"流程结束 flow_id={self._flow_id} @{ts:.1f}s")

        self._active = False
        self._flow_id = 0
        self._device_code = ""

        return flow

    def _on_voice_intent(self, msg: dict) -> None:
        """处理语音事件"""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0.0))
        key_moment = data.get("key_moment", "")
        if not key_moment:
            return

        # 不是控制关键字则视为设备识别码
        is_device = key_moment not in ["监护", "请求监护", "执行", "核对", "收到", "信息通报", "信息通告", "通报完毕", "通告完毕"]
        if is_device:
            # TODO: 等弹窗模块实现后由弹窗信号触发
            # 现在不自动启动自唱票
            # self._start_flow(ts, device_code=key_moment)
            if self._active:
                self._code_read = True

def register():
    """模块注册入口"""
    return SelfTicketRule()
