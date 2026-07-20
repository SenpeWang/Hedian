"""
监护制度

集成监护状态机，统一管理流程生命周期和状态转移。

监护制流程：
  开始：语音(监护请求) 或 行为(举手)
  内容：1.九字码复述  2.执行操作  3.核对确认
  结束：监护员和操作员离开10秒

监护绑定（key_moment）：
  当 LEADER 与操作员(ROAD1/ROAD2) 距离 ≤ BIND_DISTANCE_PX 且持续 BIND_HOLD_SEC 时
  触发绑定，保存为 key_moment。
  距离本身不是 key_moment，仅作为绑定触发条件。
"""
import json
import os
import logging

from core.event_bus import EventBus, EventTopic
from rules.rule_base import BaseRule

logger = logging.getLogger("rules.supervision")


# 操作员 → 回路名称映射（用于绑定 key_moment 文本）
_OPERATOR_LOOP_MAP = {
    "ROAD1": "一回路",
    "ROAD2": "二回路",
}


def _loop_name_from_operator(operator: str) -> str:
    """根据操作员角色名获取回路中文名称（ROAD1→一回路, ROAD2→二回路）"""
    if not operator:
        return "未知回路"
    return _OPERATOR_LOOP_MAP.get(operator, f"{operator}回路")


class SupervisionRule(BaseRule):
    """监护制度"""

    def __init__(self, config: dict = None):
        """初始化监护制度"""
        self._config = config or {}
        self._event_bus = None

        # 流程状态
        self._active = False
        self._flow_id = 0
        self._flow_counter = 0
        self._flow_start_sec = 0
        self._flow_start_source = ""
        self._target_role = None

        # 内容检查清单（由规则引擎检测的监护制关键动作）
        # code_repeat: 九字码复述是否完成（语音中出现两次相同九字码 = 操作人读 + 监护人复述）
        # execution: 执行命令是否下达（语音中出现"执行"关键词）
        # verification: 核对确认是否完成（语音中出现"核对"或"核实"关键词）
        self._checklist = {
            "code_repeat": False,   # 九字码复述
            "execution": False,     # 执行命令
            "verification": False,  # 核对确认
        }

        # 状态机状态
        self._sm_state = "IDLE"  # IDLE, REQUESTING, BOUND
        self._close_start_ts = -1.0
        self._far_start_ts = -1.0
        self._bind_hold_sec = self._config.get("bind_hold_sec", 10.0)
        self._unbind_hold_sec = self._config.get("unbind_hold_sec", 10.0)

        # 举手相关事件追踪，用于判定 5 秒内 举手 + 请求监护
        self._last_hand_raise_ts = -999.0
        self._last_hand_raise_role = None

        # 操纵人员的最新距离状态
        self._operator_states = {}

    def name(self) -> str:
        """制度名称"""
        return "supervision"

    def subscribe_events(self, event_bus: EventBus) -> None:
        """订阅事件"""
        self._event_bus = event_bus
        event_bus.subscribe(EventTopic.VOICE_KEY_MOMENT, self._on_voice_intent)
        event_bus.subscribe(EventTopic.BEHAVIOR_HAND_RAISED, self._on_mot_request)
        event_bus.subscribe(EventTopic.TRACKER_PROXIMITY, self._on_mot_status)

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
        self._sm_state = "REQUESTING"

        self._close_start_ts = -1.0
        self._far_start_ts = -1.0
        self._flow_id = self._next_flow_id()
        self._flow_start_sec = ts
        self._flow_start_source = source
        self._target_role = None  # 启动时不进行默认绑定，通过后续距离动态判断
        self._operator_states = {}
        self._checklist = {
            "code_repeat": False,
            "execution": False,
            "verification": False,
        }

        if self._event_bus:
            self._event_bus.publish(EventTopic.FLOW_STARTED, {
                "flow_id": self._flow_id,
                "flow_type": "supervision",
                "flow_start_sec": ts,
                "start_source": source,
                "target_role": None,
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

        if self._event_bus:
            self._event_bus.publish(EventTopic.FLOW_ENDED, flow, ts=ts)

        logger.info(f"流程结束 flow_id={self._flow_id} @{ts:.1f}s")

        self._active = False
        self._sm_state = "IDLE"
        self._flow_id = 0
        self._target_role = None
        self._close_start_ts = -1.0
        self._far_start_ts = -1.0

        return flow

    def _on_voice_intent(self, msg: dict) -> None:
        """处理语音事件"""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0.0))
        key_moment = data.get("key_moment", "")
        if not key_moment:
            return

        # 启动流程判定：匹配 ["监护", "请求监护"] 即可启动
        if not self._active:
            is_supervision_word = (key_moment in ["监护", "请求监护"])
            has_hand_raise = (abs(ts - self._last_hand_raise_ts) <= 5.0)

            if is_supervision_word:
                role = self._last_hand_raise_role if has_hand_raise else "ROAD1"
                self._start_flow(ts, source="voice", target_role=role)

        # 流程运行中状态更新
        if self._active:
            # 不是控制关键字则视为设备识别码
            is_device = key_moment not in ["监护", "请求监护", "执行", "核对", "信息通报", "信息通告", "通报完毕", "通告完毕"]
            if is_device:
                self._checklist["code_repeat"] = True
            elif key_moment == "执行":
                self._checklist["execution"] = True
            elif key_moment == "核对":
                self._checklist["verification"] = True
    def _on_mot_request(self, msg: dict) -> None:
        """处理 MOT 监护请求（举手）"""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0))
        role = data.get("operator", "ROAD1")

        # 仅记录举手时间与人员角色
        self._last_hand_raise_ts = ts
        self._last_hand_raise_role = role
        logger.debug(f"监护制: 收到举手事件 @{ts:.1f}s")

    def _on_mot_status(self, msg: dict) -> None:
        """处理 MOT 距离状态更新，实现状态机转移"""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0))
        state = data.get("state", "")
        operator = data.get("operator", "")

        # 记录每个操纵人员的最新监护状态
        self._operator_states[operator] = state

        # 只在流程活跃时处理距离状态
        if not self._active:
            return

        # 判断任一操纵人员是否处于监护中
        any_close = any(s == "监护中" for s in self._operator_states.values())
        # 绑定了特定操作员只检查该操作员；否则检查全部
        if self._target_role:
            target_far = (self._operator_states.get(self._target_role) == "未监护")
        else:
            target_far = all(s == "未监护" for s in self._operator_states.values())

        # 状态机转移逻辑
        if self._sm_state == "IDLE":
            pass

        elif self._sm_state == "REQUESTING":
            # REQUESTING → BOUND: 任一操作员"监护中"持续 _bind_hold_sec 秒
            if any_close:
                if self._close_start_ts < 0:
                    self._close_start_ts = ts
                elif ts - self._close_start_ts >= self._bind_hold_sec:
                    # 动态判定监护目标
                    self._target_role = next((op for op, s in self._operator_states.items() if s == "监护中"), None)

                    # 状态转移：REQUESTING → BOUND
                    self._sm_state = "BOUND"
                    self._far_start_ts = -1.0
                    loop_name = _loop_name_from_operator(self._target_role)
                    logger.info(f"状态转移: REQUESTING → BOUND @{ts:.1f}s, 监护对象={self._target_role}({loop_name})")

                    # 监护绑定作为 tracker 的 key_moment 通过事件流下发
                    bind_km = {
                        "localSec": round(ts, 2),
                        "key_moment": f"监护员已到位监护{loop_name}",
                        "source": "tracker",
                    }
                    if self._event_bus:
                        self._event_bus.publish(EventTopic.RULE_KEY_MOMENT, bind_km, ts=ts)
            else:
                self._close_start_ts = -1.0

        elif self._sm_state == "BOUND":
            # BOUND → IDLE: 目标操作员"未监护"持续 _unbind_hold_sec 秒
            if target_far:
                if self._far_start_ts < 0:
                    self._far_start_ts = ts
                elif ts - self._far_start_ts >= self._unbind_hold_sec:
                    # 状态转移：BOUND → IDLE
                    self._sm_state = "IDLE"
                    loop_name = _loop_name_from_operator(self._target_role)
                    logger.info(f"状态转移: BOUND → IDLE @{ts:.1f}s (人员离开超{self._unbind_hold_sec:.0f}秒)")

                    # 监护解绑作为 tracker 的 key_moment 通过事件流下发
                    unbind_km = {
                        "localSec": round(ts, 2),
                        "key_moment": f"监护员已离开监护{loop_name}",
                        "source": "tracker",
                    }
                    if self._event_bus:
                        self._event_bus.publish(EventTopic.RULE_KEY_MOMENT, unbind_km, ts=ts)

                    # 关闭流程
                    self._close_flow(ts, source="distance")
            else:
                self._far_start_ts = -1.0

    def save_results(self, result_dir: str) -> None:
        """规则层不保存 key_moment 文件（由 tracker 通过事件流接收并保存）"""
        pass




def register():
    """模块注册入口"""
    return SupervisionRule()
