"""
监护制度（重构版）

集成监护状态机，统一管理流程生命周期和状态转移。

监护制流程：
  开始：语音(监护请求) 或 行为(举手)
  内容：1.九字码复述  2.执行操作  3.核对确认
  结束：监护员和操作员离开10秒 或 实验结束
"""
import logging

from core.message_bus import MessageBus, MsgType
from regulations.regulation_base import BaseRegulation

logger = logging.getLogger("regulations.supervision")


class SupervisionRegulation(BaseRegulation):
    """监护制度（集成状态机）"""

    def __init__(self, config: dict = None):
        """
        初始化监护制度

        Args:
            config: 配置字典
        """
        self._config = config or {}
        self._bus = None

        # 流程状态
        self._active = False
        self._flow_id = 0
        self._flow_counter = 0
        self._flow_start_sec = 0
        self._flow_start_source = ""
        self._target_role = None

        # 内容检查清单
        self._checklist = {
            "code_repeat": False,
            "execution": False,
            "verification": False,
        }

        # 状态机状态
        self._sm_state = "IDLE"  # IDLE, REQUESTING, BOUND
        self._close_start_ts = -1.0  # 距离≤280px 开始时间
        self._far_start_ts = -1.0    # 距离>560px 开始时间
        self._bind_hold_sec = self._config.get("bind_hold_sec", 3.0)
        self._unbind_hold_sec = self._config.get("unbind_hold_sec", 10.0)

# 状态机已直接集成到本类中

    def name(self) -> str:
        """制度名称"""
        return "supervision"

    def subscribe_events(self, bus: MessageBus) -> None:
        """订阅事件"""
        self._bus = bus
        bus.subscribe(MsgType.VOICE_INTENT, self._on_voice_intent)
        bus.subscribe(MsgType.MOT_SUPERVISION_REQUEST, self._on_mot_request)
        bus.subscribe(MsgType.MOT_SUPERVISOR_STATUS, self._on_mot_status)

    def is_active(self) -> bool:
        """是否有活跃流程"""
        return self._active

    def get_current_flow(self) -> dict:
        """获取当前活跃流程"""
        if not self._active:
            return None
        return {
            "flow_id": self._flow_id,
            "flow_type": "supervision",
            "flow_start_sec": self._flow_start_sec,
            "start_source": self._flow_start_source,
            "target_role": self._target_role,
            "content_checklist": dict(self._checklist),
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

    def _start_flow(self, ts: float, source: str, target_role: str = None) -> None:
        """启动监护流程"""
        if self._active:
            return

        self._active = True
        self._sm_state = "REQUESTING"  # 状态机进入 REQUESTING 状态
        self._close_start_ts = -1.0
        self._far_start_ts = -1.0
        self._flow_id = self._next_flow_id()
        self._flow_start_sec = ts
        self._flow_start_source = source
        self._target_role = target_role or "ROAD1"
        self._checklist = {
            "code_repeat": False,
            "execution": False,
            "verification": False,
        }

        if self._bus:
            self._bus.publish(MsgType.FLOW_STARTED, {
                "flow_id": self._flow_id,
                "flow_type": "supervision",
                "flow_start_sec": ts,
                "start_source": source,
                "target_role": self._target_role,
            }, ts=ts)

        logger.info(f"流程开始 flow_id={self._flow_id} @{ts:.1f}s source={source}")

    def _close_flow(self, ts: float = 0, source: str = "unknown") -> dict:
        """关闭监护流程"""
        if not self._active:
            return None

        flow = {
            "flow_id": self._flow_id,
            "flow_type": "supervision",
            "flow_start_sec": self._flow_start_sec,
            "flow_end_sec": ts,
            "flow_continue_sec": round(ts - self._flow_start_sec, 2),
            "start_source": self._flow_start_source,
            "end_source": source,
            "target_role": self._target_role,
            "content_checklist": dict(self._checklist),
        }

        if self._bus:
            self._bus.publish(MsgType.FLOW_ENDED, flow, ts=ts)

        logger.info(f"流程结束 flow_id={self._flow_id} @{ts:.1f}s")

        self._active = False
        self._sm_state = "IDLE"  # 状态机回到 IDLE
        self._flow_id = 0
        self._target_role = None
        self._close_start_ts = -1.0
        self._far_start_ts = -1.0

        return flow

    def _on_voice_intent(self, msg: dict) -> None:
        """处理语音意图"""
        data = msg.get("data", {})
        intent = data.get("intent", "")
        ts = data.get("localSec", msg.get("ts", 0))

        # 处理流程逻辑
        if intent == "监护请求":
            if not self._active:
                self._start_flow(ts, source="voice")
            # 更新内容检查
            if self._active and data.get("key_moment") == "九字码":
                self._checklist["code_repeat"] = True

        elif intent == "操作指令":
            if self._active:
                self._checklist["execution"] = True

        elif intent in ("核对确认", "监护确认"):
            if self._active:
                self._checklist["verification"] = True

        elif intent == "实验结束":
            if self._active:
                self._close_flow(ts, source="voice")

        # 九字码检测
        if self._active and data.get("key_moment") == "九字码":
            self._checklist["code_repeat"] = True

    def _on_mot_request(self, msg: dict) -> None:
        """处理 MOT 监护请求（举手）"""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0))
        role = data.get("operator", "ROAD1")

        # 启动流程
        if not self._active:
            self._start_flow(ts, source="mot", target_role=role)

    def _on_mot_status(self, msg: dict) -> None:
        """处理 MOT 距离状态更新，实现状态机转移"""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0))
        distance_px = data.get("distance_px", 0)
        state = data.get("state", "")
        operator = data.get("operator", "")

        # 只在流程活跃时处理距离状态
        if not self._active:
            return

        # 状态机转移逻辑
        if self._sm_state == "IDLE":
            # IDLE 状态下不处理距离（等待语音/举手触发）
            pass

        elif self._sm_state == "REQUESTING":
            # REQUESTING → BOUND: 距离≤280px 持续3秒
            if state == "监护中":
                if self._close_start_ts < 0:
                    self._close_start_ts = ts
                elif ts - self._close_start_ts >= self._bind_hold_sec:
                    # 状态转移：REQUESTING → BOUND
                    self._sm_state = "BOUND"
                    self._far_start_ts = -1.0
                    logger.info(f"状态转移: REQUESTING → BOUND @{ts:.1f}s")

                    # 发布监护绑定事件
                    if self._bus:
                        self._bus.publish(MsgType.MOT_SUPERVISION_BOUND, {
                            "localSec": ts,
                            "operator": operator,
                            "source": "distance",
                        }, ts=ts)
            else:
                # 距离不够近，重置计时
                self._close_start_ts = -1.0

        elif self._sm_state == "BOUND":
            # BOUND → IDLE: 距离>560px 持续10秒
            if state == "未监护":
                if self._far_start_ts < 0:
                    self._far_start_ts = ts
                elif ts - self._far_start_ts >= self._unbind_hold_sec:
                    # 状态转移：BOUND → IDLE
                    self._sm_state = "IDLE"
                    logger.info(f"状态转移: BOUND → IDLE @{ts:.1f}s (人员离开超10秒)")

                    # 发布监护结束事件
                    if self._bus:
                        self._bus.publish(MsgType.MOT_SUPERVISION_END, {
                            "localSec": ts,
                            "operator": operator,
                            "source": "distance",
                            "reason": "人员离开超10秒",
                        }, ts=ts)

                    # 关闭流程
                    self._close_flow(ts, source="distance", reason="人员离开超10秒")
            else:
                # 距离够近，重置计时
                self._far_start_ts = -1.0




def register():
    """模块注册入口"""
    return SupervisionRegulation()
