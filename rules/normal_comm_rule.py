"""普通沟通流程.

触发与生命周期：
  开始：识别到岗位词（一回路/二回路/机组长）后 3s 内识别到"请讲"
  结束（三种）：
    1. 流程中识别到"收到" → 提前结束（end_source=recv_ack）；
    2. 流程中**再次**出现"岗位词 + 3s 内请讲" → 上一个普通沟通结束、
       新的普通沟通开始（end_source=new_comm）；
    3. 或开始时间点后 50s 自动结束（end_source=timeout）。

同句应答：岗位词、"请讲"、"收到"可能落在同一句话里（例如
"一回路请讲我给拿一下程序…好收到开始关"）。语音层会给出按句中位置排序的
关键词列表，"请讲"之后的"收到"视为同句应答，流程立即收尾，不空等超时。

与监护制一致，通过 FLOW_STARTED / FLOW_ENDED 事件驱动前端"系统通知"
（"普通沟通开始/结束"）与流程事件记录。

说明：50s 超时由独立计时器（threading.Timer）实现，不依赖任何 tracker 事件；
结束时间固定为 开始时间点 + 50s（视频时间轴），不受模块推理速度差影响。
"""

import logging
import threading
from typing import Any, Dict, Optional

from core.event_bus import EventBus, EventTopic
from rules.rule_base import BaseRule

logger = logging.getLogger("rules.normal_comm")

# 岗位词（触发前置条件）
POST_WORDS = ("一回路", "二回路", "机组长")
# 岗位词之后多少秒内出现"请讲"则启动流程（也是"新沟通顶掉旧流程"的判定窗口）
START_WINDOW_SEC = 3.0
# 流程最长持续秒数
MAX_FLOW_SEC = 50.0


class NormalCommRule(BaseRule):
    """普通沟通流程."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化普通沟通流程状态.

        Args:
            config: 制度配置字典，可为 None.
        """
        self._config = config or {}
        self._event_bus: Optional[EventBus] = None

        # 流程状态
        self._active = False
        self._flow_id = 0
        self._flow_counter = 0
        self._flow_start_sec = 0.0
        self._flow_start_source = ""

        # 待启动状态：最近一次岗位词时间
        self._last_post_word_ts = -999.0

        # MAX_FLOW_SEC 超时计时器（独立线程，不依赖 tracker 事件）
        self._timeout_timer: Optional[threading.Timer] = None

    def name(self) -> str:
        """制度名称.

        Returns:
            制度名称字符串, 恒为 "normal_comm".
        """
        return "normal_comm"

    def subscribe_events(self, event_bus: EventBus) -> None:
        """订阅普通沟通流程关心的事件.

        Args:
            event_bus: 事件总线.
        """
        self._event_bus = event_bus
        event_bus.subscribe(EventTopic.VOICE_KEY_MOMENT, self._on_voice_intent)

    # ---------------- 流程生命周期 ----------------
    def _start_flow(self, timestamp: float, source: str) -> None:
        """启动普通沟通流程.

        Args:
            timestamp: 流程开始时间（秒）.
            source: 触发来源（如 voice）.
        """
        if self._active:
            return

        self._active = True
        self._flow_counter += 1
        self._flow_id = self._flow_counter
        self._flow_start_sec = timestamp
        self._flow_start_source = source
        # 消费掉本次配对用到的岗位词记忆：避免同一句里的重复"请讲"反复触发，
        # 也避免流程结束后 3s 内再来一个"请讲"被误认为新沟通。
        self._last_post_word_ts = -999.0

        if self._event_bus:
            self._event_bus.publish(
                EventTopic.FLOW_STARTED,
                {
                    "flow_id": self._flow_id,
                    "flow_type": "normal_comm",
                    "flow_start_sec": timestamp,
                    "start_source": source,
                },
                timestamp=timestamp,
            )

        logger.info(
            f"普通沟通流程开始 flow_id={self._flow_id} @{timestamp:.1f}s "
            f"source={source}"
        )

        # 超时计时器：到点自动结束，结束时间固定为 开始时间点 + MAX_FLOW_SEC（视频时间轴）
        self._cancel_timer()
        self._timeout_timer = threading.Timer(
            MAX_FLOW_SEC, self._on_timer_timeout
        )
        self._timeout_timer.daemon = True
        self._timeout_timer.start()

    def _on_timer_timeout(self) -> None:
        """计时器到点回调：按 开始时间点 + MAX_FLOW_SEC 关闭流程."""
        if not self._active:
            return
        self._close_flow(self._flow_start_sec + MAX_FLOW_SEC, source="timeout")

    def _close_flow(self, timestamp: float, source: str) -> None:
        """关闭普通沟通流程并发布 FLOW_ENDED 事件.

        Args:
            timestamp: 流程结束时间（秒）.
            source: 结束来源（recv_ack=识别到收到 / new_comm=被新的普通沟通顶掉 /
                timeout=超过 MAX_FLOW_SEC）.
        """
        if not self._active:
            return

        self._cancel_timer()

        flow = {
            "flow_id": self._flow_id,
            "flow_type": "normal_comm",
            "flow_start_sec": self._flow_start_sec,
            "flow_end_sec": timestamp,
            "flow_continue_sec": round(timestamp - self._flow_start_sec, 2),
            "start_source": self._flow_start_source,
            "end_source": source,
        }

        if self._event_bus:
            self._event_bus.publish(
                EventTopic.FLOW_ENDED, flow, timestamp=timestamp
            )

        logger.info(
            f"普通沟通流程结束 flow_id={self._flow_id} @{timestamp:.1f}s "
            f"持续 {flow['flow_continue_sec']}s source={source}"
        )

        self._active = False
        self._flow_id = 0
        # 注意：此处**不能**清空 _last_post_word_ts。
        # 若流程是被"超时"收尾的，而触发收尾的那条事件恰好是下一次普通沟通的
        # 岗位词（已在 _on_voice_intent 里登记），清空会让紧随其后的"请讲"
        # 找不到前置岗位词而无法启动新流程（实测缺陷：13:44"机组长请讲"失效）。
        # 岗位词记忆改由 _start_flow 在"消费配对"时清空。

    # ---------------- 事件处理 ----------------
    def _cancel_timer(self) -> None:
        """取消未触发的超时计时器."""
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self._timeout_timer = None

    def _on_voice_intent(self, event: Dict[str, Any]) -> None:
        """处理语音关键事件：启动 / 被新沟通顶掉 / "收到"提前结束 / 超时收尾.

        判定顺序（顺序本身是关键，勿随意调整）：

          1. **岗位词登记**：无论当前是否处于流程中都要登记。流程中再次出现
             岗位词，很可能意味着新的普通沟通即将发起（见第 3 条）；
          2. **"收到"提前结束**（流程运行中且仍在 MAX_FLOW_SEC 窗口内）：
             按收到时刻结束（end_source=recv_ack）；
          3. **超时收尾**（流程运行中且时间戳已超出窗口）：按
             开始 + MAX_FLOW_SEC 关闭（end_source=timeout）。
             **此处不 return**（方案A）——触发收尾的这条事件本身可能正是下一次
             普通沟通的岗位词，若在此丢弃，紧随其后的"请讲"将找不到前置岗位词
             而无法启动新流程；
          4. **被新沟通顶掉**（流程仍然活跃）：窗口内出现"岗位词 + 3s 内请讲"
             → 结束当前流程（end_source=new_comm）并立即开启新的普通沟通；
          5. **未处于流程中**：按"岗位词 + 3s 内请讲"启动（含同句应答处理）。

        同句应答：若"请讲"所在句的后续关键词里出现"收到"（`keys` 已按句中出现
        位置排序），视为对方在同一句内应答，启动后立即收尾，不空等超时。

        Args:
            event: 总线消息字典, 业务载荷取其 "data" 字段.
        """
        payload = event.get("data", {})
        timestamp = float(payload.get("localSec", event.get("ts", 0.0)))
        key_moment = payload.get("key_moment", "")
        if not key_moment:
            return

        # 1) 岗位词登记：不受"是否处于流程中"限制
        if key_moment in POST_WORDS:
            self._last_post_word_ts = timestamp
            logger.debug(
                f"普通沟通: 登记岗位词 {key_moment} @{timestamp:.1f}s"
                f"（当前{'流程中' if self._active else '空闲'}）"
            )

        if self._active:
            # 2) "收到"提前结束（仅在窗口内有效；超出窗口交由第 3 条按超时收尾）
            if (
                key_moment == "收到"
                and timestamp - self._flow_start_sec <= MAX_FLOW_SEC
            ):
                self._close_flow(timestamp, source="recv_ack")
                return

            # 3) 超时收尾：不 return，让同一条事件继续参与第 4/5 步的判定
            if timestamp - self._flow_start_sec >= MAX_FLOW_SEC:
                self._close_flow(
                    self._flow_start_sec + MAX_FLOW_SEC, source="timeout"
                )

            # 4) 被新的普通沟通顶掉：岗位词后 START_WINDOW_SEC 内出现"请讲"
            if self._active and key_moment == "请讲":
                gap = timestamp - self._last_post_word_ts
                if 0 <= gap <= START_WINDOW_SEC:
                    logger.info(
                        f"普通沟通: 岗位词后 {gap:.1f}s 出现“请讲”，"
                        f"结束 flow_id={self._flow_id} 并开启新的普通沟通"
                    )
                    self._close_flow(timestamp, source="new_comm")
                    self._start_flow(timestamp, source="voice")
                    return

        # 5) 未处于流程中：岗位词后 START_WINDOW_SEC 内出现"请讲"则启动
        if not self._active and key_moment == "请讲":
            gap = timestamp - self._last_post_word_ts
            if 0 <= gap <= START_WINDOW_SEC:
                self._start_flow(timestamp, source="voice")
                # 同句应答：岗位词/"请讲"/"收到"可能在同一句话里，此时"收到"事件
                # 的排序与时间戳都与"请讲"一致，不能因为三者同句就丢掉该确认信号，
                # 否则流程只能空等到 MAX_FLOW_SEC 超时才结束。
                if self._has_ack_after(payload.get("keys"), key_moment):
                    logger.info(
                        f"普通沟通: 请讲 @{timestamp:.1f}s 同句内检测到“收到”，"
                        "立即确认结束（同句应答）"
                    )
                    self._close_flow(timestamp, source="recv_ack")
            else:
                logger.debug(
                    f"普通沟通: 请讲 @{timestamp:.1f}s 距岗位词 {gap:.1f}s 超出窗口，不启动"
                )

    @staticmethod
    def _has_ack_after(keys: Optional[Any], current_key: str) -> bool:
        """判断同一句的关键词序列中，"收到"是否出现在当前词之后.

        语音层给出的 `keys` 已按关键词在句中的实际出现位置排序，因此可直接用下标
        比较先后。仅当"收到"确实排在当前词之后时才算同句应答；若"收到"出现在
        之前（例如上一轮对话的残留），不得误判为本次应答。

        Args:
            keys: 该句识别到的关键词列表（已按位置排序），可为 None.
            current_key: 当前正在处理的关键词（如 "请讲"）.

        Returns:
            "收到" 出现在 current_key 之后返回 True.
        """
        if not keys or not isinstance(keys, (list, tuple)):
            return False
        ordered = list(keys)
        try:
            index = ordered.index(current_key)
        except ValueError:
            # 当前词不在列表中（异常数据）：只要本句含"收到"即按同句应答处理
            return "收到" in ordered
        tail_start = index + 1
        return "收到" in ordered[tail_start:]

    def reset(self) -> None:
        """重置普通沟通状态，防止新一轮推理混入旧流程状态."""
        super().reset()
        self._cancel_timer()
        self._active = False
        self._flow_id = 0
        self._flow_start_sec = 0.0
        self._last_post_word_ts = -999.0

    def save_results(self, result_dir: str) -> None:
        """规则层不保存产物（流程事件由 FlowEventRecorder 统一记录）.

        Args:
            result_dir: 结果目录(本制度不产出文件, 该参数不使用).
        """
        pass


def register() -> NormalCommRule:
    """实例化并返回普通沟通流程实例.

    Returns:
        可注册到注册表的普通沟通流程实例.
    """
    return NormalCommRule()
