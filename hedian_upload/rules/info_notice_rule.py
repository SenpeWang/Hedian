"""信息通报制度.

按 docs/rule_doc/信息通报.md 的规程实现 BaseRule 状态机，对主控室"信息通报"流程做在线
合规检测，并在流程结束时发布 FLOW_ENDED 交给评估层（Qwen3-8B）打分。

规程五步（团队级判定，不区分一回路 / 二回路）：
    (1) 信息发起者举手 + 高声喊出"信息通报"/"信息通告"（红线：单独举手不启动流程）
    (2) 主控室其他成员听到后立刻停下手中工作接受信息
    (3) 团队成员均予以关注，发起人方可进行后续的信息传递
    (4) 信息传递结束，发起者喊出"通报完毕"/"通告完毕"
    (5) 值长（US）回答"收到"予以回应

检测口径（与 evaluation/qwen_evaluator._build_info_notice_prompt 的评估维度一一对应）：
    - 步骤(1)：voice 关键事件 + 举手事件的双向 ±HAND_RAISE_WINDOW_SEC 时间窗。仅有语音
      时流程照样启动（供评估按"仅语音发起"扣分），同时明确记录红线违规。"高声"维度当前
      无音量输入，以语音关键事件作为可检测代理。
    - 步骤(2)(3)：由 GazeModule 的"信息通报 10 秒关注度窗口"给出结论——本制度发布
      FLOW_STARTED(flow_type=info_notice) 后，gaze 模块开启 10s 注视采样窗口，窗口结束
      下发 GAZE_ATTENTION(has_turned) 并落盘"已给予关注/没有给予关注"。
    - 步骤(4)：即时闭环——喊出"通报完毕"立即关闭流程，不等待"收到"回复。
    - 步骤(5)："收到"通常晚于"通报完毕"到达，故闭环后再多留 ACK_EVIDENCE_TAIL_SEC 的
      取证尾巴（evidence_end_sec）供数据提取，流程时长仍按喊话时刻计。

实时子步骤下发（"系统通知"面板逐行显示）：
    每项判定在观测当场经 RULE_FLOW_STEP 下发一行（含判定时刻与合规与否），不做"等流程
    收口再汇总"的批处理，因此面板呈现为随时间推进的事件流，例如：
        [00:55] 信息通报开始
        [01:00] 未举手
        [00:55] 回复请讲
        [01:05] 停下工作并予以关注
    仍未观测到的项在流程收口时统一补发"未…"行。

证据到齐再收口（跨模块时序）：
    voice（ASR）处理速度约为画面推理的 3 倍以上，tracker / gaze 逐帧推理远慢于它。规则内部
    时钟由"收到事件的最大 localSec"推进，若据此到点就直接收口，gaze 的 10 秒关注度结论
    （画面 65s，墙钟要晚数十秒才送达）会被判成"未关注"。故本制度收口前先挂起：超时、待闭环
    到期或被下一次发起覆盖时，若关注度结论尚未到达，就把该流程的状态快照压入挂起队列并从
    活跃状态摘除（后续流程照常发起），等它自己那条结论（按 flow_id 回填）到达、或等满
    EVIDENCE_WAIT_SEC（墙钟，视频提前结束由 finalize 兜底）后再真正收口。收口时用快照复现
    现场，步骤补齐、事件发布、面板时序与正常收口完全一致。
"""
import logging
import time
from typing import Any, Dict, List, Optional

from core.event_bus import EventBus, EventTopic
from rules.rule_base import BaseRule

logger = logging.getLogger("rules.info_notice")

# 语音关键事件标签（voice 模块已把"信息通告/通告完毕"归一化为下列标准标签，
# 此处同时接受原始写法，便于直接对接上游事件）
NOTICE_START_MOMENTS = ("信息通报", "信息通告")
NOTICE_END_MOMENTS = ("通报完毕", "通告完毕")
NOTICE_ACK_MOMENT = "收到"
NOTICE_PLEASE_MOMENT = "请讲"

#: 评估维度（团队级）：(判定键名, 中文名, 单项分值)。五项合计 10 分
TEAM_CHECKLIST_ITEMS = (
    ("raise_hand_and_shout", "举手并喊出信息通报", 2),
    ("replied_please_speak", "回复请讲", 2),
    ("stopped_and_attended", "停下手中工作并予以关注", 2),
    ("shout_finished", "喊出通报完毕", 2),
    ("replied_received", "回答收到", 2),
)

#: 满分（五项 × 2 分）
TOTAL_SCORE = sum(weight for _key, _label, weight in TEAM_CHECKLIST_ITEMS)

#: 实时步骤标识（"系统通知"面板按事件时刻逐行显示，不等到流程收口）
STEP_HAND_RAISE = "hand_raise"
STEP_PLEASE_SPEAK = "please_speak"
STEP_ATTEND = "attend"
STEP_FINISHED = "finished"
STEP_RECEIVED = "received"

#: 语音关键事件 → (实时步骤标识, 判定键名, 面板文案)
VOICE_STEP_TABLE = {
    NOTICE_PLEASE_MOMENT: (STEP_PLEASE_SPEAK, "replied_please_speak", "回复请讲"),
    NOTICE_ACK_MOMENT: (STEP_RECEIVED, "replied_received", "回答收到"),
}


class InfoNoticeRule(BaseRule):
    """信息通报制度."""

    #: 举手与"信息通报"语音的双向配合时间窗（秒）：举手落在语音前后该窗口内即视为配合
    HAND_RAISE_WINDOW_SEC = 5.0

    #: "通报完毕"即时闭环后，仍纳入评估取证的时间长度（秒）：值长"收到"通常晚于闭环到达
    ACK_EVIDENCE_TAIL_SEC = 10.0

    #: 流程超时兜底（秒）：超过该时长仍未喊出"通报完毕"则强制闭环，避免流程悬挂
    MAX_FLOW_SEC = 180.0

    #: 收口前等待未决证据（关注度结论）的墙钟上限（秒）：慢模块逐帧推理远慢于 ASR，
    #: 到点即收口会把"还没算出来的关注度"判成未关注
    EVIDENCE_WAIT_SEC = 60.0

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化信息通报制度.

        Args:
            config: 制度配置字典（config.yaml 的 info_notice 段），可为 None.
        """
        self._config = config or {}
        self._event_bus: Optional[EventBus] = None

        self._hand_raise_window_sec = float(
            self._config.get("hand_raise_window_sec", self.HAND_RAISE_WINDOW_SEC))
        self._ack_evidence_tail_sec = float(
            self._config.get("ack_evidence_tail_sec", self.ACK_EVIDENCE_TAIL_SEC))
        self._max_flow_sec = float(
            self._config.get("max_flow_sec", self.MAX_FLOW_SEC))
        self._evidence_wait_sec = float(
            self._config.get("evidence_wait_sec", self.EVIDENCE_WAIT_SEC))

        self._active = False
        self._flow_id = 0
        self._flow_counter = 0
        self._flow_start_sec = 0.0
        self._flow_start_source = "voice"

        # 最近一次举手；身份由 tracker 赋予，缺失时保持 None，不自行赋予
        self._last_hand_raise_ts = -999.0
        self._last_hand_raise_role: Optional[str] = None

        # 最近一次观测到的事件时间戳：供 finalize 收尾时关闭流程使用
        # （避免用 0 关闭导致提取时间窗为空、评估拿不到任何证据）
        self._last_event_ts = 0.0

        self._initiator_role: Optional[str] = None
        self._attention_checked = False
        self._attention_check_sec = -1.0
        self._checklist = self._empty_checklist()

        #: 本轮流程窗口内收集到的语音回应事件 [{label, sec}]
        self._voice_acks: list = []
        #: 待闭环（"通报完毕"已喊出，仍在等"收到"落地）: (end_sec, source)
        self._pending_close: Optional[tuple] = None
        #: 收口挂起队列：每项是一个流程的状态快照（见 _capture_flow_state），
        #: 等其关注度结论到达、或墙钟超期后由 _release_hold 收口。用队列而非单槽：
        #: voice 远快于逐帧推理，前一个流程挂着等结论时后续流程往往已经发起。
        self._holds: list = []

        # ---- 实时步骤下发状态（"系统通知"面板逐行显示，判定当场推送）----
        #: 已下发步骤的集合（STEP_* 常量）
        self._emitted_steps: set = set()
        #: 举手判定截止时刻（flow_start + 窗口），到期仍未举手即下发"未举手"
        self._hand_raise_deadline: Optional[float] = None
        #: 举手步骤结论：True=已下发合规发起，False=已下发未举手，None=尚未定论
        self._hand_raise_resolved: Optional[bool] = None

    def name(self) -> str:
        """制度名称.

        Returns:
            制度标识，用于 config.yaml 的 rules 段开关与注册表键.
        """
        return "info_notice"

    def apply_config(self, config: Dict[str, Any]) -> None:
        """回灌 config.yaml 的 info_notice 配置段.

        规则实例由 RuleRegistry.discover() 无参构造，配置只能在订阅事件前回灌；
        仅覆盖显式给出的键，未给出的保持默认值。

        Args:
            config: 配置字典（config.yaml 的 info_notice 段），可为 None/空.
        """
        self._config = {**self._config, **(config or {})}
        self._hand_raise_window_sec = float(
            self._config.get("hand_raise_window_sec", self.HAND_RAISE_WINDOW_SEC))
        self._ack_evidence_tail_sec = float(
            self._config.get("ack_evidence_tail_sec", self.ACK_EVIDENCE_TAIL_SEC))
        self._max_flow_sec = float(
            self._config.get("max_flow_sec", self.MAX_FLOW_SEC))
        self._evidence_wait_sec = float(
            self._config.get("evidence_wait_sec", self.EVIDENCE_WAIT_SEC))
        logger.info(
            f"信息通报: 配置已应用 超时={self._max_flow_sec:.0f}s "
            f"回应窗={self._ack_evidence_tail_sec:.0f}s "
            f"证据等待={self._evidence_wait_sec:.0f}s")

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

    # ------------------------------------------------------------------
    # 事件回调
    # ------------------------------------------------------------------
    def _on_hand_raised(self, event: Dict[str, Any]) -> None:
        """处理举手事件（BEHAVIOR_HAND_RAISED，payload={localSec, operator, track_id?}）.

        红线规则：单独举手不启动流程，本回调只记录举手时刻与身份，供"信息通报"语音
        到达时做 ±窗口配合判定。举手落在语音前后窗口内均视为配合。

        Args:
            event: 事件总线分发的标准事件字典.
        """
        payload = event.get("data", {})
        timestamp = payload.get("localSec", event.get("ts", 0))

        self._last_hand_raise_ts = timestamp
        self._last_hand_raise_role = payload.get("operator")
        logger.debug(
            f"信息通报: 收到举手事件 @{timestamp:.1f}s（单独举手不触发信息通报流程）")
        # 先补判再推进时钟：否则窗口边界上的举手会被 _tick 抢先下发成"未举手"
        self._upgrade_on_late_hand_raise(timestamp)
        # 已进入"收口挂起"的流程同样补判：语音远快于画面，举手常在挂起期间才送达
        self._upgrade_held_flows(timestamp)
        self._tick(timestamp)

    def _on_voice_intent(self, event: Dict[str, Any]) -> None:
        """处理语音关键事件（VOICE_KEY_MOMENT，payload={localSec, key_moment}）.

        Args:
            event: 事件总线分发的标准事件字典.
        """
        payload = event.get("data", {})
        key_moment = payload.get("key_moment", "")
        timestamp = payload.get("localSec", event.get("ts", 0.0))
        # 先推进内部时钟：超时兜底依赖"已观测到的最大事件时间戳"，与本次事件是否命中
        # 关键词无关（voice 的普通语句同样代表画面/推理已推进到该时刻）
        self._tick(timestamp)
        if not key_moment:
            return

        if key_moment in NOTICE_START_MOMENTS:
            if self._active:
                # 上一轮未喊"通报完毕"就再次发起：先按被覆盖闭环，保证流程成对出现。
                # 关注度结论未到时走挂起队列，其迟到结论仍会回填到该流程，避免把
                # 尚未算出的关注度直接判成"未关注"。
                logger.warning(
                    f"信息通报: 流程 flow_id={self._flow_id} 未结束即再次发起，按被覆盖关闭")
                self._request_close(timestamp, source="superseded")
            self._start_flow(timestamp, source="voice")
        elif key_moment in NOTICE_END_MOMENTS:
            if self._active and self._pending_close is None:
                self._checklist["shout_finished"] = True
                self._emit_step(timestamp, "喊出通报完毕", True, STEP_FINISHED)
                # 不立即闭环：团队成员与值长的"收到"通常紧随"通报完毕"到达，
                # 若此刻就收摊，"回答收到"的判定将永远拿不到证据。
                # 改为"记下结束时刻 + 延后 ACK_EVIDENCE_TAIL_SEC 再发布"，流程时长
                # 仍按喊话时刻计，发布延迟对评估层无影响（本就等待取证末端）。
                self._pending_close = (timestamp, "normal_end")
                logger.info(
                    f"信息通报: 发起人喊出'通报完毕' @{timestamp:.1f}s，"
                    f"等待 {self._ack_evidence_tail_sec:.0f}s 内的'收到'回应后闭环")
        elif key_moment in (NOTICE_PLEASE_MOMENT, NOTICE_ACK_MOMENT):
            if self._active:
                self._voice_acks.append(
                    {"label": key_moment, "sec": round(timestamp, 2)})
                # 判定即时定论，面板当场显示；收口时不再重新归属，避免前后矛盾
                self._emit_ack_step(timestamp, key_moment)
                logger.info(
                    f"信息通报: 观测到回应'{key_moment}' @{timestamp:.1f}s")

    def _on_gaze_status(self, event: Dict[str, Any]) -> None:
        """处理关注度结论事件（GAZE_ATTENTION，payload={flow_id, localSec, has_turned}）.

        GazeModule 在收到 FLOW_STARTED(flow_type=info_notice) 后为该 flow_id 开启
        10 秒注视采样窗口，窗口结束时下发本事件并落盘"已给予关注/没有给予关注"。
        结论按 flow_id 归属：命中挂起队列的回填该流程后再释放挂起，命中当前活跃流程
        的当场生效；两者都不是（更早轮次的残留）则忽略，避免污染当前流程判定。

        Args:
            event: 事件总线分发的标准事件字典.
        """
        payload = event.get("data", {})
        timestamp = payload.get("localSec", event.get("ts", 0))
        has_turned = bool(payload.get("has_turned", False))
        flow_id = payload.get("flow_id")

        if flow_id is not None and int(flow_id) >= 0:
            for hold in self._holds:
                if hold["flow_id"] != int(flow_id):
                    continue
                # 挂起流程的迟到结论：回填快照，交由 _release_hold 收口
                hold["checklist"]["stopped_and_attended"] = has_turned
                hold["attention_checked"] = True
                hold["attention_check_sec"] = round(
                    timestamp - hold["flow_start_sec"], 2)
                hold["late_attention"] = {
                    "sec": float(timestamp), "has_turned": has_turned}
                logger.info(
                    f"信息通报: flow_id={flow_id} 关注度结论迟到 @{timestamp:.1f}s，"
                    f"回填挂起中的流程后收口")
                self._tick(timestamp)
                return

        self._tick(timestamp)
        if not self._active:
            return
        if flow_id is not None and int(flow_id) >= 0 \
                and int(flow_id) != self._flow_id:
            logger.info(
                f"信息通报: 忽略非本流程的关注度结论 flow_id={flow_id}"
                f"（当前 flow_id={self._flow_id}）")
            return

        self._attention_checked = True
        self._attention_check_sec = round(timestamp - self._flow_start_sec, 2)
        self._checklist["stopped_and_attended"] = has_turned
        # 关注度窗口结束即定论，面板即时下发"停下工作并予以关注 / 未停下工作并予以关注"
        self._emit_step(
            timestamp,
            "停下工作并予以关注" if has_turned else "未停下工作并予以关注",
            has_turned, STEP_ATTEND)

        if has_turned:
            logger.info(
                f"信息通报: 团队成员已给予关注 @{timestamp:.1f}s"
                f"（启动后 {self._attention_check_sec:.1f}s）")
        else:
            logger.warning(
                f"信息通报: 团队成员未给予关注 @{timestamp:.1f}s，发起人不应继续信息传递")

        # 关注度结论是收口前最后一个待齐证据：挂起中的收口在此释放
        # （上面的 _tick 先执行、此时结论尚未落库，故需在其后补一次检查）
        self._release_hold()

    # ------------------------------------------------------------------
    # 状态机
    # ------------------------------------------------------------------
    def _upgrade_on_late_hand_raise(self, timestamp: float) -> None:
        """用后到的举手事件补判步骤(1)（跨模块时序补偿）."""
        if self._active and not self._checklist["raise_hand_and_shout"] \
                and abs(timestamp - self._flow_start_sec) <= self._hand_raise_window_sec:
            self._checklist["raise_hand_and_shout"] = True
            self._initiator_role = self._last_hand_raise_role
            logger.info(f"信息通报: 举手事件后到 @{timestamp:.1f}s，补判为合规发起")
            role_suffix = f"（{self._initiator_role}）" if self._initiator_role else ""
            self._emit_step(timestamp, f"举手发起{role_suffix}", True,
                            STEP_HAND_RAISE)
            self._hand_raise_resolved = True

    def _upgrade_held_flows(self, timestamp: float) -> None:
        """用后到的举手事件补判"收口挂起中"的流程（跨模块时序补偿）.

        语音模块把事件时钟推得远快于逐帧推理，流程常因超时先被压入挂起队列
        （_request_close 后已 _detach_active_flow，_active=False），此时举手事件才
        随画面推理送达，_upgrade_on_late_hand_raise 的 _active 前提已不成立。若不做
        这一层补判，画面时间明明落在窗口内的举手仍会被定格成"未举手"（红线误判）。

        Args:
            timestamp: 举手事件的画面时间戳（秒）.
        """
        for hold in self._holds:
            if hold["checklist"].get("raise_hand_and_shout"):
                continue
            if abs(timestamp - hold["flow_start_sec"]) > self._hand_raise_window_sec:
                continue
            hold["checklist"]["raise_hand_and_shout"] = True
            hold["hand_raise_resolved"] = True
            if not hold.get("initiator_role"):
                hold["initiator_role"] = self._last_hand_raise_role
            hold["late_hand_raise"] = {
                "sec": float(timestamp),
                "role": self._last_hand_raise_role,
            }
            logger.info(
                f"信息通报: flow_id={hold['flow_id']} 举手事件迟到 @{timestamp:.1f}s"
                f"（画面时间落在窗口内），补判为合规发起，随收口生效")

    def _tick(self, timestamp: float) -> None:
        """推进内部时钟并做超时兜底.

        Args:
            timestamp: 当前事件时间戳（秒）.
        """
        if timestamp > self._last_event_ts:
            self._last_event_ts = timestamp

        # 举手配合窗口到期仍未见到举手：当场下发"未举手"（红线违规的实时提示）
        if self._active and self._hand_raise_resolved is None \
                and self._hand_raise_deadline is not None \
                and timestamp >= self._hand_raise_deadline:
            self._emit_step(self._hand_raise_deadline, "未举手", False,
                            STEP_HAND_RAISE)
            self._hand_raise_resolved = False

        # 待闭环流程：等"通报完毕"后 ACK_EVIDENCE_TAIL_SEC 内的回应收集完毕再发布
        if self._active and self._pending_close is not None:
            end_sec, source = self._pending_close
            if timestamp >= end_sec + self._ack_evidence_tail_sec:
                self._pending_close = None
                self._request_close(end_sec, source)

        if self._active \
                and timestamp - self._flow_start_sec > self._max_flow_sec:
            # 按"开始 + 时限"这一固定边界收口，而不是按触发收口的那个偶发事件时间：
            # 后者会让流程时长取决于下一个事件何时到达（曾出现 180s 时限却闭环出
            # 293.68s），并把评估取证窗口一起拖到视频末端，白等数分钟。
            timeout_sec = self._flow_start_sec + self._max_flow_sec
            logger.warning(
                f"信息通报: 流程超时未结束（>{self._max_flow_sec:.0f}s），按时限 "
                f"{timeout_sec:.1f}s 收口 flow_id={self._flow_id}")
            self._request_close(timeout_sec, "timeout")

        # 挂起中的收口：证据到齐（或等待超期）即真正收口
        self._release_hold()

    def _request_close(self, end_sec: float, source: str) -> None:
        """请求收口：证据未到齐时先挂起，等关注度结论到达再收口.

        voice 事件把内部时钟推得远快于画面推理，到点直接收口会把 gaze 尚未产出的
        关注度结论判成"未关注"。此处改为把流程快照压入挂起队列（上限
        EVIDENCE_WAIT_SEC 墙钟）并立即从活跃状态摘除：后续流程照常可以发起，
        被挂起的流程等自己那条关注度结论（按 flow_id 回填）到达后再收口。

        Args:
            end_sec: 流程结束时刻（秒）.
            source: 结束来源（normal_end/timeout/superseded/finalize）.
        """
        if not self._active:
            return
        if self._attention_checked:
            self._close_flow(end_sec, source=source)
            return
        if any(hold["flow_id"] == self._flow_id for hold in self._holds):
            # 同一流程已有挂起请求（先到的收口条件决定结束时刻），不覆盖
            return
        snapshot = self._capture_flow_state()
        snapshot.update({
            "end_sec": end_sec,
            "source": source,
            "deadline": time.monotonic() + self._evidence_wait_sec,
            "late_attention": None,
        })
        self._holds.append(snapshot)
        logger.warning(
            f"信息通报: 收口条件已满足（{source} @{end_sec:.1f}s），但关注度结论未到，"
            f"挂起 flow_id={self._flow_id} 最多等待 {self._evidence_wait_sec:.0f}s 后收口")
        self._detach_active_flow()

    def _release_hold(self) -> None:
        """释放收口挂起队列：关注度结论已到或等待超期时执行真正的收口."""
        for hold in list(self._holds):
            if not (hold["attention_checked"]
                    or time.monotonic() >= hold["deadline"]):
                continue
            self._holds.remove(hold)
            if hold["attention_checked"]:
                logger.info(
                    f"信息通报: flow_id={hold['flow_id']} 关注度结论已到，结束挂起并收口")
            else:
                logger.warning(
                    f"信息通报: flow_id={hold['flow_id']} 等待关注度结论超期，"
                    f"按原定时刻收口")
            self._close_flow_from_snapshot(hold)

    def _capture_flow_state(self) -> Dict[str, Any]:
        """抓取当前流程的完整状态（挂起快照与收口后还原现场共用）.

        Returns:
            状态字典：流程标识、判定清单、已下发步骤，以及跨流程共享的举手观测。
        """
        return {
            "flow_id": self._flow_id,
            "flow_start_sec": self._flow_start_sec,
            "flow_start_source": self._flow_start_source,
            "initiator_role": self._initiator_role,
            "attention_checked": self._attention_checked,
            "attention_check_sec": self._attention_check_sec,
            "checklist": dict(self._checklist),
            "voice_acks": [dict(ack) for ack in self._voice_acks],
            "emitted_steps": set(self._emitted_steps),
            "hand_raise_deadline": self._hand_raise_deadline,
            "hand_raise_resolved": self._hand_raise_resolved,
            "last_hand_raise_ts": self._last_hand_raise_ts,
            "last_hand_raise_role": self._last_hand_raise_role,
        }

    def _apply_flow_state(self, state: Dict[str, Any]) -> None:
        """把状态字典装回状态机.

        Args:
            state: _capture_flow_state 产出的状态字典.
        """
        self._active = True
        self._flow_id = state["flow_id"]
        self._flow_start_sec = state["flow_start_sec"]
        self._flow_start_source = state["flow_start_source"]
        self._initiator_role = state["initiator_role"]
        self._attention_checked = state["attention_checked"]
        self._attention_check_sec = state["attention_check_sec"]
        self._checklist = dict(state["checklist"])
        self._voice_acks = [dict(ack) for ack in state["voice_acks"]]
        self._emitted_steps = set(state["emitted_steps"])
        self._hand_raise_deadline = state["hand_raise_deadline"]
        self._hand_raise_resolved = state["hand_raise_resolved"]
        self._last_hand_raise_ts = state["last_hand_raise_ts"]
        self._last_hand_raise_role = state["last_hand_raise_role"]
        self._pending_close = None

    def _detach_active_flow(self) -> None:
        """清空活跃流程状态（快照已另存），不发布任何事件."""
        self._active = False
        self._flow_id = 0
        self._flow_start_sec = 0.0
        self._flow_start_source = "voice"
        self._initiator_role = None
        self._attention_checked = False
        self._attention_check_sec = -1.0
        self._checklist = self._empty_checklist()
        self._voice_acks = []
        self._pending_close = None
        self._emitted_steps = set()
        self._hand_raise_deadline = None
        self._hand_raise_resolved = None
        self._last_hand_raise_ts = -999.0
        self._last_hand_raise_role = None

    def _close_flow_from_snapshot(
        self, hold: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """用挂起快照收口某个流程.

        挂起与收口可能发生在"另一个流程正活跃"的时刻（voice 远快于逐帧推理，多个
        流程紧邻发起），故不能直接改当前状态机：先保存现场，装回快照收口，事后再
        还原——这样步骤补齐、事件发布、评估提交全部复用 _close_flow 这一条路径。

        Args:
            hold: 挂起队列中的快照（含 end_sec / source / late_attention）.

        Returns:
            关闭的流程事件字典；该流程已无判定清单时为 None.
        """
        saved = self._capture_flow_state() if self._active else None
        self._apply_flow_state(hold)
        try:
            late = hold.get("late_attention")
            if late is not None:
                # 挂起期间迟到的关注度结论：先补下面板判定行，再随收口一起发布
                self._emit_step(
                    late["sec"],
                    "停下工作并予以关注" if late["has_turned"]
                    else "未停下工作并予以关注",
                    late["has_turned"], STEP_ATTEND)
            late_raise = hold.get("late_hand_raise")
            if late_raise is not None:
                # 挂起期间迟到的举手：窗口到期时推送的"未举手"行已发出、无法撤回，
                # 此处在其后补一行补判结果（同一时刻、排在"未举手"之下），
                # 使面板口径与最终评分一致（STEP_HAND_RAISE 已被占用，先解除去重）
                role_suffix = (f"（{late_raise['role']}）"
                               if late_raise.get("role") else "")
                emit_sec = hold.get("hand_raise_deadline") or (
                    hold["flow_start_sec"] + self._hand_raise_window_sec)
                self._emitted_steps.discard(STEP_HAND_RAISE)
                self._emit_step(emit_sec, f"举手发起{role_suffix}（补判）", True,
                                STEP_HAND_RAISE)
            return self._close_flow(hold["end_sec"], source=hold["source"],
                                    silent=bool(hold.get("silent")))
        finally:
            if saved is not None:
                self._apply_flow_state(saved)
            else:
                self._detach_active_flow()

    def _start_flow(self, timestamp: float, source: str) -> None:
        """启动信息通报流程并发布 FLOW_STARTED（gaze 模块据此开启 10s 关注度窗口）.

        Args:
            timestamp: 流程开始时间（秒）.
            source: 流程触发来源（如 voice）.
        """
        # 双向时间窗：举手落在"信息通报"语音前后 HAND_RAISE_WINDOW_SEC 内即视为配合
        has_hand_raise = (
            abs(timestamp - self._last_hand_raise_ts) <= self._hand_raise_window_sec)
        initiator_role = self._last_hand_raise_role if has_hand_raise else None

        self._active = True
        self._flow_id = self._next_flow_id()
        self._flow_start_sec = timestamp
        self._flow_start_source = source
        self._last_event_ts = max(self._last_event_ts, timestamp)
        self._initiator_role = initiator_role
        self._attention_checked = False
        self._attention_check_sec = -1.0
        self._checklist = self._empty_checklist()
        self._checklist["raise_hand_and_shout"] = has_hand_raise
        self._voice_acks = []
        self._pending_close = None
        # 注意：不动 _holds。挂起队列里可能还押着上一个流程（等它自己的关注度结论），
        # 新流程的开始不应打断那一笔收口。
        # 实时步骤状态随新流程清零；举手先记"待判定 + 截止时刻"
        self._emitted_steps = set()
        self._hand_raise_resolved = None
        self._hand_raise_deadline = timestamp + self._hand_raise_window_sec

        # FLOW_STARTED 用于通知 gaze 模块开启 10s 关注度窗口
        # 红线（未举手）不在此下定论：语音远快于 tracker，举手事件常在"信息通报"
        # 之后才到达，此处判定会把合规发起误判为违规。判定见 _upgrade_on_late_hand_raise，
        # 最终以 behavior/tracker 落盘的举手关键事件为准（评估层据此打分）。
        if self._event_bus:
            self._event_bus.publish(EventTopic.FLOW_STARTED, {
                "flow_id": self._flow_id,
                "flow_type": "info_notice",
                "flow_start_sec": timestamp,
                "start_source": source,
                "initiator_role": initiator_role,
            },
                                    timestamp=timestamp)

        # 举手若在启动前已观测到，当场下发合规发起；否则留给窗口到期/补判两条路径
        if has_hand_raise:
            role_suffix = f"（{initiator_role}）" if initiator_role else ""
            self._emit_step(timestamp, f"举手发起{role_suffix}", True,
                            STEP_HAND_RAISE)
            self._hand_raise_resolved = True

        logger.info(
            f"信息通报流程开始 flow_id={self._flow_id} @{timestamp:.1f}s "
            f"举手配合={has_hand_raise} 发起人={initiator_role} source={source}")

    def _close_flow(
        self, timestamp: float = 0, source: str = "unknown",
        silent: bool = False
    ) -> Optional[Dict[str, Any]]:
        """关闭信息通报流程并发布 FLOW_ENDED 事件.

        Args:
            timestamp: 流程结束时间（秒）；非正值时回退到最近一次观测时刻.
            source: 结束来源（normal_end/timeout/superseded/finalize）.
            silent: 静默收口。True 表示这不是"观测到的真实结束"，而是兜底收口
                （上一轮被中断、或视频结束时流程仍活跃）。此时不下发面板步骤行，
                结束事件也带 silent 标记供前端跳过，避免上一轮残留流程的行落进
                新一轮已清空的面板，出现"开始/结束同一时刻"的自相矛盾显示。
                内部状态清理与持久化照常执行，评估不受影响。

        Returns:
            流程事件字典；无活跃流程时返回 None.
        """
        if not self._active:
            return None

        end_sec = timestamp if timestamp > 0 else self._last_event_ts
        # 事件时间戳缺失或乱序时，保证结束时间不早于开始时间，避免提取窗口为空
        end_sec = max(end_sec, self._flow_start_sec)

        # 收口前补齐未观测到的步骤（逐行下发"未…"），确保面板与最终评分口径一致；
        # 必须在 FLOW_ENDED 之前下发，前端才能按时间序排在"结束"行之前。
        # 静默收口跳过：兜底收口的"未…"行若推给面板，会污染新一轮已清空的列表。
        if not silent:
            self._flush_pending_steps(end_sec)

        # 面板展示用结束时刻：normal_end 时流程时长按喊话时刻计，但"收到"回应落在其后的
        # 取证尾巴里。若面板也按喊话时刻记"结束"，就会出现"结束"之后还有回应行的错觉，
        # 故展示时刻取取证尾巴结束的时刻；其他收口方式本就即时，与 end_sec 一致。
        panel_end_sec = (end_sec + self._ack_evidence_tail_sec
                         if source == "normal_end" else end_sec)

        flow = {
            "flow_id": self._flow_id,
            "flow_type": "info_notice",
            "flow_start_sec": round(self._flow_start_sec, 2),
            "flow_end_sec": round(end_sec, 2),
            "flow_continue_sec": round(end_sec - self._flow_start_sec, 2),
            # 即时闭环后仍向外多看一段：值长"收到"晚于闭环到达，需落在提取区间内
            "evidence_end_sec": round(end_sec + self._ack_evidence_tail_sec, 2),
            "panel_end_sec": round(panel_end_sec, 2),
            "start_source": self._flow_start_source,
            "end_source": source,
            #: 兜底收口标记：前端据此跳过本行的面板展示
            "silent": silent,
            "initiator_role": self._initiator_role,
            "attention_checked": self._attention_checked,
            "attention_check_sec": self._attention_check_sec,
            "content_checklist": dict(self._checklist),
            # ---- 团队级评分（5 项 × 2 分 = 10 分，不区分回路）----
            "team_total_score": self._team_score(),
            "team_max_score": TOTAL_SCORE,
            "voice_ack_events": [dict(ack) for ack in self._voice_acks],
        }

        if self._event_bus:
            self._event_bus.publish(EventTopic.FLOW_ENDED, flow,
                                    timestamp=end_sec)

        logger.info(
            f"信息通报流程结束 flow_id={self._flow_id} @{end_sec:.1f}s source={source} "
            f"得分={flow['team_total_score']}/{TOTAL_SCORE} 清单={flow['content_checklist']}")

        self._active = False
        self._flow_id = 0
        self._initiator_role = None
        self._last_hand_raise_ts = -999.0
        self._pending_close = None
        self._voice_acks = []
        self._emitted_steps = set()
        self._hand_raise_deadline = None
        self._hand_raise_resolved = None

        return flow

    def finalize(
        self, final_timestamp: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """视频结束收尾：关闭挂起队列中与仍活跃的流程.

        优先用 _pending_close 的精确收口时刻（"通报完毕"落地时间）；若没有，再用调用方
        传入的 final_timestamp（视频结束时刻）兜底；都没有才退回 _last_event_ts。

        一律走静默收口：finalize 是"没等到流程按规程自然收口"的兜底路径，它产生的
        结束时刻只是"最后观测时刻"而非真实结束，若推给面板就会和新一轮的"开始"行
        挤在同一时刻显示，用户看到的就是"开始/结束同时发生"。内部收口与持久化照常。

        Args:
            final_timestamp: 视频结束时刻（秒），由 main.on_pipeline_done 从启动期
              读到的视频最大时长传入。无 _pending_close 时优先使用，避免回退到
              _last_event_ts 在 voice 快 / tracker 慢时远小于真实结束时刻.

        Returns:
            最后关闭的流程事件字典；没有任何待收口流程时返回 None.
        """
        closed_flows: List[Dict[str, Any]] = []
        for hold in list(self._holds):
            self._holds.remove(hold)
            hold["silent"] = True
            logger.warning(
                f"信息通报: 视频结束，flow_id={hold['flow_id']} 挂起等待中的流程"
                f"按已定时刻静默收尾")
            closed = self._close_flow_from_snapshot(hold)
            if closed is not None:
                closed_flows.append(closed)

        if self._active:
            if self._pending_close is not None:
                end_sec, source = self._pending_close
                logger.warning("信息通报: 视频结束，待闭环流程按'通报完毕'时刻静默收尾")
                closed = self._close_flow(timestamp=end_sec, source=source, silent=True)
            else:
                if final_timestamp is not None:
                    logger.warning(
                        f"信息通报: 视频结束仍有活跃流程，按视频结束时刻 {final_timestamp:.1f}s 静默收尾"
                    )
                    end_sec = final_timestamp
                else:
                    logger.warning("信息通报: 视频结束仍有活跃流程，按最后观测时刻静默收尾")
                    end_sec = self._last_event_ts
                closed = self._close_flow(timestamp=end_sec,
                                          source="finalize", silent=True)
            if closed is not None:
                closed_flows.append(closed)

        return closed_flows[-1] if closed_flows else None

    def reset(self) -> None:
        """重置信息通报状态，防止新一轮推理混入旧的举手/关注/流程状态."""
        super().reset()
        self._active = False
        self._flow_id = 0
        self._flow_start_sec = 0.0
        self._flow_start_source = "voice"
        self._last_hand_raise_ts = -999.0
        self._last_hand_raise_role = None
        self._last_event_ts = 0.0
        self._initiator_role = None
        self._attention_checked = False
        self._attention_check_sec = -1.0
        self._checklist = self._empty_checklist()
        self._voice_acks = []
        self._pending_close = None
        self._holds.clear()
        self._emitted_steps = set()
        self._hand_raise_deadline = None
        self._hand_raise_resolved = None

    def _team_score(self) -> int:
        """计算团队总分（五项判定 × 单项分值）.

        Returns:
            团队总分（满分 TOTAL_SCORE）.
        """
        return sum(
            weight for key, _label, weight in TEAM_CHECKLIST_ITEMS
            if self._checklist.get(key)
        )

    # ------------------------------------------------------------------
    # 实时步骤下发（"系统通知"面板逐行显示，判定当场推送）
    # ------------------------------------------------------------------
    def _emit_step(self, sec: float, text: str, ok: bool, step: str) -> None:
        """下发一条流程子步骤判定（判定当场推送，不等流程收口）.

        Args:
            sec: 该判定的逻辑时刻（秒）；用于前端与画面同步显示，须不晚于当前事件时刻.
            text: 面板展示文案（如"未举手"/"回复请讲"）.
            ok: 判定结果，True=合规，False=违规.
            step: 步骤标识（STEP_* 常量），同一流程内同一步骤只下发一次.
        """
        if step in self._emitted_steps:
            return
        self._emitted_steps.add(step)

        if self._event_bus:
            self._event_bus.publish(EventTopic.RULE_FLOW_STEP, {
                "flow_id": self._flow_id,
                "flow_type": "info_notice",
                "step": step,
                "ok": bool(ok),
                "text": text,
                "step_sec": round(float(sec), 2),
            }, timestamp=float(sec))
        logger.info(f"信息通报: 步骤下发[{text}] ok={ok} @{sec:.1f}s")

    def _emit_ack_step(self, sec: float, label: str) -> None:
        """把一条语音回应判定为合规并下发步骤（团队级：不区分回路）.

        Args:
            sec: 语音回应时刻（秒）.
            label: 语音关键事件标签（"请讲"/"收到"）.
        """
        entry = VOICE_STEP_TABLE.get(label)
        if not entry:
            return
        step, key, action = entry
        self._checklist[key] = True
        self._emit_step(sec, action, True, step)

    def _flush_pending_steps(self, end_sec: float) -> None:
        """收口前补齐所有未观测到的步骤（逐条下发违规行）.

        Args:
            end_sec: 流程结束时刻（秒），作为补发步骤的显示时刻.
        """
        if self._hand_raise_resolved is None:
            # 走到这里说明窗口到期前始终没见到举手（合规发起会在 _start_flow /
            # _upgrade_on_late_hand_raise 里当场下发并把该标记置 True）
            self._emit_step(end_sec, "未举手", False, STEP_HAND_RAISE)
            self._hand_raise_resolved = False

        for step, key, action in (
            (STEP_PLEASE_SPEAK, "replied_please_speak", "回复请讲"),
            (STEP_ATTEND, "stopped_and_attended", "停下工作并予以关注"),
            (STEP_RECEIVED, "replied_received", "回答收到"),
        ):
            if not self._checklist.get(key):
                self._emit_step(end_sec, f"未{action}", False, step)

        if not self._checklist.get("shout_finished"):
            self._emit_step(end_sec, "未喊出通报完毕", False, STEP_FINISHED)

    def _empty_checklist(self) -> Dict[str, Any]:
        """构造初始检查清单（对应规程五个评估维度，团队级）.

        Returns:
            检查清单字典，键与 TEAM_CHECKLIST_ITEMS 一致；全部为 False 表示尚未达成.
        """
        return {key: False for key, _label, _weight in TEAM_CHECKLIST_ITEMS}


def register() -> InfoNoticeRule:
    """实例化并返回信息通报制度实例.

    Returns:
        可注册到注册表的信息通报制度实例.
    """
    return InfoNoticeRule()
