"""
监护制度（重构版）

集成监护状态机，统一管理流程生命周期和状态转移。

监护制流程：
  开始：语音(监护请求) 或 行为(举手)
  内容：1.九字码复述  2.执行操作  3.核对确认
  结束：监护员和操作员离开10秒
"""
import json
import os
import logging

from core.event_bus import EventBus, EventTopic
from rules.rule_base import BaseRule

logger = logging.getLogger("rules.supervision")


class SupervisionRule(BaseRule):
    """监护制度（集成状态机）"""

    def __init__(self, config: dict = None):
        """
        初始化监护制度

        Args:
            config: 配置字典
        """
        self._config = config or {}
        self._event_bus = None

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

        # 举手相关事件追踪，用于判定 5 秒内 举手 + 请求监护
        self._last_hand_raise_ts = -999.0
        self._last_hand_raise_role = None
        self._events = []  # 保存监护事件(BOUND/UNBOUND)

        # 记录操纵人员的最新距离状态，不进行默认绑定
        self._operator_states = {}

# 状态机已直接集成到本类中

    def name(self) -> str:
        """制度名称"""
        return "supervision"

    def subscribe_events(self, event_bus: EventBus) -> None:
        """订阅事件"""
        self._event_bus = event_bus
        event_bus.subscribe(EventTopic.VOICE_KEY_MOMENT, self._on_voice_intent)
        event_bus.subscribe(EventTopic.TRACKER_HAND_RAISED, self._on_mot_request)
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
        self._sm_state = "REQUESTING"  # 状态机进入 REQUESTING 状态

        # 设置 Redis 标志，通知其他模块监护制已开始
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        r.set("supervision:active", "true", ex=3600)
        r.close()
        self._close_start_ts = -1.0
        self._far_start_ts = -1.0
        self._flow_id = self._next_flow_id()
        self._flow_start_sec = ts
        self._flow_start_source = source
        self._target_role = None  # 启动时不进行默认绑定，通过后续身份距离动态判断
        self._operator_states = {}  # 清空历史状态记录
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
        self._sm_state = "IDLE"  # 状态机回到 IDLE
        self._flow_id = 0
        self._target_role = None
        self._close_start_ts = -1.0
        self._far_start_ts = -1.0

        # 清除 Redis 标志，通知其他模块监护制已结束
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        r.set("supervision:active", "false", ex=3600)
        r.close()

        return flow

    def _on_voice_intent(self, msg: dict) -> None:
        """处理语音事件（仅包含 localSec 和 key_moment 字段）"""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0.0))
        key_moment = data.get("key_moment", "")
        if not key_moment:
            return

        # 启动流程判定：只要匹配到包含“监护”二字的语音事件即可启动流程 (支持“由监护进入”以及“监护和举手一起进入”)
        if not self._active:
            is_supervision_word = (key_moment in ["监护", "请求监护"])
            has_hand_raise = (abs(ts - self._last_hand_raise_ts) <= 5.0)

            if is_supervision_word:
                role = self._last_hand_raise_role if has_hand_raise else "ROAD1"
                self._start_flow(ts, source="voice", target_role=role)

        # 流程运行中状态更新
        if self._active:
            # 如果 km 不是核心的控制关键字，它就是设备识别码！
            is_device = key_moment not in ["监护", "请求监护", "执行", "核对", "收到", "信息通报", "信息通告", "通报完毕", "通告完毕"]
            if is_device:
                self._checklist["code_repeat"] = True
            elif key_moment == "执行":
                self._checklist["execution"] = True
            elif key_moment == "核对" or key_moment == "收到":
                self._checklist["verification"] = True
    def _on_mot_request(self, msg: dict) -> None:
        """处理 MOT 监护请求（举手）"""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0))
        role = data.get("operator", "ROAD1")

        # 仅记录举手时间与人员角色，单独举手不再判定/启动监护流程
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

        # 判断任一操纵人员是否处于监护中，以及被监护的操作员是否离开
        any_close = any(s == "监护中" for s in self._operator_states.values())
        # 如果绑定了特定操作员，只检查该操作员；否则检查全部
        if self._target_role:
            target_far = (self._operator_states.get(self._target_role) == "未监护")
        else:
            target_far = all(s == "未监护" for s in self._operator_states.values())

        # 状态机转移逻辑
        if self._sm_state == "IDLE":
            pass

        elif self._sm_state == "REQUESTING":
            # REQUESTING → BOUND: 任一操纵员距离≤280px 持续3秒
            if any_close:
                if self._close_start_ts < 0:
                    self._close_start_ts = ts
                elif ts - self._close_start_ts >= self._bind_hold_sec:
                    # 动态判定当前的监护目标是谁（获取第一个处于监护中的操纵员）
                    self._target_role = next((op for op, s in self._operator_states.items() if s == "监护中"), None)

                    # 状态转移：REQUESTING → BOUND
                    self._sm_state = "BOUND"
                    self._far_start_ts = -1.0
                    logger.info(f"状态转移: REQUESTING → BOUND @{ts:.1f}s, 动态监护对象: {self._target_role}")

                    # 保存绑定事件到JSON
                    operator_name = self._target_role or "未知"
                    self._events.append({
                        "localSec": round(ts, 2),
                        "key_moment": f"监护员和{operator_name}建立监护关系",
                        "source": "tracker",
                    })
                    # 发布监护绑定事件
                    if self._event_bus:
                        self._event_bus.publish(EventTopic.TRACKER_SUPERVISION_BOUND, {
                            "localSec": ts,
                            "operator": self._target_role,
                            "source": "distance",
                        }, ts=ts)
            else:
                self._close_start_ts = -1.0

        elif self._sm_state == "BOUND":
            # BOUND → IDLE: 全部操纵员离开(距离>560px) 持续10秒
            if all_far:
                if self._far_start_ts < 0:
                    self._far_start_ts = ts
                elif ts - self._far_start_ts >= self._unbind_hold_sec:
                    # 状态转移：BOUND → IDLE
                    self._sm_state = "IDLE"
                    logger.info(f"状态转移: BOUND → IDLE @{ts:.1f}s (人员离开超10秒)")

                    operator_name = self._target_role or "未知"
                    self._events.append({
                        "localSec": round(ts, 2),
                        "key_moment": f"监护员和{operator_name}解除监护关系",
                        "source": "tracker",
                    })
                    # 发布监护结束事件
                    if self._event_bus:
                        self._event_bus.publish(EventTopic.TRACKER_SUPERVISION_END, {
                            "localSec": ts,
                            "operator": self._target_role,
                            "source": "distance",
                            "reason": "人员离开超10秒",
                        }, ts=ts)

                    # 关闭流程
                    self._close_flow(ts, source="distance", reason="人员离开超10秒")
            else:
                self._far_start_ts = -1.0




def register():
    """模块注册入口"""
    return SupervisionRule()
