"""
监护制度.

集成监护状态机，统一管理流程生命周期和状态转移。

监护制流程：
  开始：语音"请求监护"即启动（5 秒内伴随举手则指定举手者为监护对象）
  内容：1.九字码复述  2.执行操作  3.核对确认
  结束：监护员和操作员离开10秒

监护绑定（key_moment）：
  当 LEADER 与操作员(ROAD1/ROAD2) 距离 ≤ BIND_DISTANCE_PX 且持续 BIND_HOLD_SEC 时
  触发绑定，保存为 key_moment。
  距离本身不是 key_moment，仅作为绑定触发条件。
"""
import logging

from core.event_bus import EventBus, EventTopic
from rules.rule_base import BaseRule

logger = logging.getLogger("rules.supervision")


class SupervisionRule(BaseRule):
    """监护制度."""

    def __init__(self, config: dict = None):
        """初始化监护制度."""
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
        # code_repeat: 九字码复述  execution: 执行命令  verification: 核对确认
        # finger_screen/finger_file: 手指指向屏幕/文件（启动后由事件填充）
        self._checklist = {
            "code_repeat": False,  # 九字码复述
            "execution": False,  # 执行命令
            "verification": False,  # 核对确认
        }

        # 状态机: IDLE → REQUESTING → BOUND → IDLE
        self._sm_state = "IDLE"
        self._close_start_ts = -1.0  # REQUESTING 中"监护中"连续计时起点
        self._far_start_ts = -1.0  # BOUND 中"未监护"连续计时起点
        self._bind_hold_sec = self._config.get("bind_hold_sec", 10.0)
        self._unbind_hold_sec = self._config.get("unbind_hold_sec", 10.0)

        # 最近举手（时间+角色），供启动时指定监护对象
        self._last_hand_raise_ts = -999.0
        self._last_hand_raise_role = None

        # 各操作员最新距离状态 + 是否曾绑定
        self._operator_states = {}
        self._ever_bound = False

    def name(self) -> str:
        """制度名称."""
        return "supervision"

    def subscribe_events(self, event_bus: EventBus) -> None:
        """订阅事件."""
        self._event_bus = event_bus
        event_bus.subscribe(EventTopic.VOICE_KEY_MOMENT, self._on_voice_intent)
        event_bus.subscribe(EventTopic.BEHAVIOR_HAND_RAISED,
                            self._on_mot_request)
        event_bus.subscribe(EventTopic.BEHAVIOR_FINGER_SCREEN,
                            self._on_finger_screen)
        event_bus.subscribe(EventTopic.BEHAVIOR_FINGER_FILE,
                            self._on_finger_file)
        event_bus.subscribe(EventTopic.TRACKER_PROXIMITY, self._on_mot_status)

    def _start_flow(self,
                    ts: float,
                    source: str,
                    target_role: str = None) -> None:
        """启动监护流程."""
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
        self._ever_bound = False
        self._checklist = {
            "code_repeat": False,
            "execution": False,
            "verification": False,
            "finger_screen": False,
            "finger_file": False,
        }

        if self._event_bus:
            self._event_bus.publish(EventTopic.FLOW_STARTED, {
                "flow_id": self._flow_id,
                "flow_type": "supervision",
                "flow_start_sec": ts,
                "start_source": source,
                "target_role": None,
            },
                                    ts=ts)

        logger.info(f"流程开始 flow_id={self._flow_id} @{ts:.1f}s source={source}")

    def _close_flow(self, ts: float = 0, source: str = "unknown") -> dict:
        """关闭监护流程."""
        if not self._active:
            return None

        is_supervised = getattr(self, "_ever_bound", False) or (self._sm_state
                                                                == "BOUND")

        # 若流程结束时从未到位监护，下发“未监护” key_moment
        if not is_supervised:
            no_bind_km = {
                "localSec": round(ts, 2),
                "key_moment": "监护员未到位（未监护）",
                "source": "tracker",
            }
            if self._event_bus:
                self._event_bus.publish(EventTopic.RULE_KEY_MOMENT,
                                        no_bind_km,
                                        ts=ts)
            logger.info(f"监护流程结束 @{ts:.1f}s: 监护员未到位（未监护）")

        flow = {
            "flow_id": self._flow_id,
            "flow_type": "supervision",
            "flow_start_sec": self._flow_start_sec,
            "flow_end_sec": ts,
            "flow_continue_sec": round(ts - self._flow_start_sec, 2),
            "start_source": self._flow_start_source,
            "end_source": source,
            "target_role": self._target_role,
            "is_supervised": is_supervised,
            "supervision_status": "已监护" if is_supervised else "未监护",
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
        """处理语音事件."""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0.0))
        key_moment = data.get("key_moment", "")
        if not key_moment:
            return

        # 启动流程判定：语音"监护"/"请求监护"即启动，5秒内举手则指定监护对象
        if not self._active:
            is_supervision_word = (key_moment in ["监护", "请求监护"])
            has_hand_raise = (abs(ts - self._last_hand_raise_ts) <= 5.0)

            if is_supervision_word:
                # 5秒内有举手则用举手者身份，否则不指定（后续通过距离动态绑定）
                role = self._last_hand_raise_role if has_hand_raise else None
                self._start_flow(ts, source="voice", target_role=role)

        # 流程运行中：非控制关键字视为设备识别码（九字码）
        if self._active:
            is_device = key_moment not in [
                "监护", "请求监护", "执行", "核对", "信息通报", "信息通告", "通报完毕", "通告完毕"
            ]
            if is_device:
                self._checklist["code_repeat"] = True
            elif key_moment == "执行":
                self._checklist["execution"] = True
            elif key_moment == "核对":
                self._checklist["verification"] = True

    def _on_finger_screen(self, msg: dict) -> None:
        """处理手指指向屏幕事件."""
        if self._active:
            self._checklist["finger_screen"] = True
            logger.info("监护制: 记录手指指向屏幕操作")

    def _on_finger_file(self, msg: dict) -> None:
        """处理手指指向文件事件（有程序分支关键特征."""
        if self._active:
            self._checklist["finger_file"] = True
            logger.info("监护制: 记录手指指向文件操作(有程序分支关键特征)")

    def _on_mot_request(self, msg: dict) -> None:
        """处理 MOT 监护请求（举手."""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0))
        # 身份由跟踪模块赋予后随事件下发；缺失时保留 None，不自行赋予
        role = data.get("operator")

        # 仅记录举手时间与人员角色
        self._last_hand_raise_ts = ts
        self._last_hand_raise_role = role
        logger.debug(f"监护制: 收到举手事件 @{ts:.1f}s")

    def _on_mot_status(self, msg: dict) -> None:
        """处理 MOT 距离状态更新，实现状态机转移."""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0))
        state = data.get("state", "")
        operator = data.get("operator", "")

        # 记录每个操纵人员的最新监护状态
        self._operator_states[operator] = state

        # 流程超时兜底（300 秒无进展自动关闭）
        if self._active and ts - self._flow_start_sec > 300.0:
            logger.warning(f"监护制流程超时未结束，自动关闭 flow_id={self._flow_id}")
            self._close_flow(ts, source="timeout")
            return

        if not self._active:
            return

        any_close = any(s == "监护中" for s in self._operator_states.values())
        # 已绑定则只检查目标操作员，否则检查全部
        if self._target_role:
            target_far = (self._operator_states.get(
                self._target_role) == "未监护")
        else:
            target_far = all(s == "未监护"
                             for s in self._operator_states.values())

        # 状态机转移
        if self._sm_state == "IDLE":
            pass

        elif self._sm_state == "REQUESTING":
            # REQUESTING → BOUND: "监护中"持续 bind_hold_sec
            if any_close:
                if self._close_start_ts < 0:
                    self._close_start_ts = ts
                elif ts - self._close_start_ts >= self._bind_hold_sec:
                    self._target_role = next(
                        (op for op, s in self._operator_states.items()
                         if s == "监护中"), None)
                    self._sm_state = "BOUND"
                    self._ever_bound = True
                    self._far_start_ts = -1.0
                    loop_name = self._target_role or ""
                    logger.info(
                        f"状态转移: REQUESTING → BOUND @{ts:.1f}s, 监护对象={self._target_role}({loop_name})"
                    )

                    # 下发"监护员已到位" key_moment
                    bind_km = {
                        "localSec": round(ts, 2),
                        "key_moment": f"监护员已到位监护{loop_name}",
                        "source": "tracker",
                    }
                    if self._event_bus:
                        self._event_bus.publish(EventTopic.RULE_KEY_MOMENT,
                                                bind_km,
                                                ts=ts)
            else:
                self._close_start_ts = -1.0

        elif self._sm_state == "BOUND":
            # BOUND → IDLE: 目标操作员"未监护"持续 unbind_hold_sec
            if target_far:
                if self._far_start_ts < 0:
                    self._far_start_ts = ts
                elif ts - self._far_start_ts >= self._unbind_hold_sec:
                    self._sm_state = "IDLE"
                    loop_name = self._target_role or ""
                    logger.info(
                        f"状态转移: BOUND → IDLE @{ts:.1f}s (人员离开超{self._unbind_hold_sec:.0f}秒)"
                    )

                    # 下发"监护员已离开" key_moment 并关闭流程
                    unbind_km = {
                        "localSec": round(ts, 2),
                        "key_moment": f"监护员已离开监护{loop_name}",
                        "source": "tracker",
                    }
                    if self._event_bus:
                        self._event_bus.publish(EventTopic.RULE_KEY_MOMENT,
                                                unbind_km,
                                                ts=ts)
                    self._close_flow(ts, source="distance")
            else:
                self._far_start_ts = -1.0

    def reset(self) -> None:
        """重置监护制状态，防止新一轮推理混入旧的人员/流程状态."""
        super().reset()
        self._checklist = {
            "code_repeat": False,
            "execution": False,
            "verification": False,
            "finger_screen": False,
            "finger_file": False,
        }
        self._operator_states = {}
        self._last_hand_raise_ts = -999.0
        self._last_hand_raise_role = None
        self._ever_bound = False

    def save_results(self, result_dir: str) -> None:
        """规则层不保存 key_moment 文件（由 tracker 通过事件流接收并保存."""
        pass


def register():
    """模块注册入口."""
    return SupervisionRule()
