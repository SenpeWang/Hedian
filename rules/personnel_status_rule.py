"""人员状态监控制度.

实现 BaseRule 接口，管理人员状态监控规程。

包含三条规则：
1. 监控操纵员，确保 2 人中至少 1 人监控机组，当两名操纵员视线同时
   离开盘台超过 1 分钟时，进行一次记录。
2. 监视主控室人数，当人员少于 1 人时，进行一次记录。
3. 监视工位在岗：身份模块一旦确认离岗（口径为 identity.away_hold_sec），
   即记一条离岗违规，返岗时回填实际时长。
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from core.event_bus import EventBus, EventTopic
from rules.rule_base import BaseRule

logger = logging.getLogger("rules.personnel_status")

# 工位角色中文名（用于违规文案）
_IDENTITY_CN = {"LEADER": "机组长", "ROAD1": "操作员1", "ROAD2": "操作员2"}
# 违规明细文件名（写入 result_dir/rules/ 下）
VIOLATIONS_FILENAME = "personnel_status_violations.json"


class PersonnelStatusRule(BaseRule):
    """人员状态监控制度."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化人员状态监控制度.

        Args:
            config: `rules.params.personnel_status` 配置段，可为 None；
                缺省值即规程口径（视线离开盘台 60s、无人值守 10s）.
        """
        self._config = config or {}
        self._event_bus: Optional[EventBus] = None

        # 阈值全部来自配置（落盘可改），代码不留魔法数
        self._gaze_away_sec = float(self._config.get("gaze_away_sec", 60.0))
        self._no_people_sec = float(self._config.get("no_people_sec", 10.0))

        self._violations: List[Dict[str, Any]] = []
        self._current_people_count = -1

        self._no_people_start_ts = None
        self._last_event_ts = 0.0
        # 工位离岗：离岗口径完全交给 identity.away_hold_sec（连续离开工位
        # 多边形超过该秒数即确认离岗），本制度不再叠加第二道时长阈值
        self._away_start: Dict[str, float] = {}
        # 身份 -> 尚未回填时长的离岗违规条目（away_end 时补时长）
        self._open_violation: Dict[str, Dict[str, Any]] = {}
        self._result_dir: Optional[str] = None

    def name(self) -> str:
        """制度名称.

        Returns:
            制度名称字符串.
        """
        return "personnel_status"

    def subscribe_events(self, event_bus: EventBus) -> None:
        """订阅本制度关心的事件.

        Args:
            event_bus: 事件总线.
        """
        self._event_bus = event_bus
        event_bus.subscribe(EventTopic.GAZE_ALERT, self._on_gaze_alert)
        event_bus.subscribe(
            EventTopic.TRACKER_HEADCOUNT, self._on_people_count
        )
        # 工位离岗（identity 模块判定并发布）
        event_bus.subscribe(EventTopic.IDENTITY_AWAY, self._on_identity_away)

    def set_result_dir(self, result_dir: str) -> None:
        """设置结果目录（违规明细落盘位置）.

        Args:
            result_dir: 本轮结果目录.
        """
        self._result_dir = result_dir

    def is_active(self) -> bool:
        """人员状态监控始终是激活状态.

        Returns:
            恒为 True，人员状态监控无流程开关.
        """
        return True

    def finalize(self) -> Optional[Dict[str, Any]]:
        """收尾：检查未结束的无人值守状态是否超过告警阈值.

        Returns:
            无返回内容，违规记录直接追加到 _violations.
        """
        # 视频收尾时，检查是否仍有未结束的无人值守状态且时长已达阈值
        if self._no_people_start_ts is not None:
            duration_sec = self._last_event_ts - self._no_people_start_ts
            if duration_sec >= self._no_people_sec:
                description = (
                    f"警报：主控室无人值守（人员少于1人）持续超过"
                    f"{self._no_people_sec:g}S（实际持续 {duration_sec:.1f}秒）"
                )
                logger.warning(
                    f"人员状态违规 (PeopleCount/Finalize): {description} "
                    f"@{self._last_event_ts:.1f}s"
                )
                violation = {
                    "localSec": self._no_people_start_ts,
                    "event": "NO_OPERATOR_VIOLATION",
                    "count": 0,
                    "duration": round(duration_sec, 2),
                    "source": "tracker",
                    "description": description,
                }
                self._violations.append(violation)
            self._no_people_start_ts = None

        # 收尾时仍有未闭合的离岗：回填时长（违规本身在 away_start 已记录）
        for identity, violation in list(self._open_violation.items()):
            start = self._away_start.get(identity)
            if start is None:
                continue
            duration = max(0.0, self._last_event_ts - start)
            cn = _IDENTITY_CN.get(identity, identity)
            violation["duration"] = round(duration, 2)
            violation["description"] = (
                f"警报：{cn}离岗持续 {duration:.1f}秒（视频结束仍未返岗）"
            )
            self._open_violation.pop(identity, None)
            self._away_start.pop(identity, None)
        return None

    def _on_gaze_alert(self, event: Dict[str, Any]) -> None:
        """处理两名操纵员视线同时离开盘台超过 60 秒的事件.

        Args:
            event: 事件载荷, 含 localSec 与 away_duration、heads_count 等判定字段.
        """
        payload = event.get("data", {})
        timestamp = payload.get("localSec", event.get("ts", 0))
        self._last_event_ts = max(self._last_event_ts, timestamp)
        away_duration = payload.get("away_duration", 0)
        heads_count = payload.get("heads_count", 0)

        # 只有持续达到阈值（规程口径 60 秒）才记录违规
        if away_duration < self._gaze_away_sec:
            logger.debug(
                f"凝视告警忽略: 仅 {away_duration:.1f}秒，"
                f"未达 {self._gaze_away_sec:g} 秒阈值"
            )
            return

        description = (
            f"警报：两名操纵员视线同时离开盘台持续超过"
            f"{self._gaze_away_sec:g}秒（实际持续 {away_duration:.1f}秒）"
        )
        logger.warning(f"人员状态违规 (Gaze): {description} @{timestamp:.1f}s")

        violation = {
            "localSec": timestamp,
            "event": "PANEL_VIOLATION",
            "away_duration": away_duration,
            "heads_count": heads_count,
            "source": "gaze",
            "description": description,
        }
        self._violations.append(violation)

    def _on_people_count(self, event: Dict[str, Any]) -> None:
        """处理人数统计事件（TRACKER_HEADCOUNT）.

        Args:
            event: 事件载荷, 含 localSec 与 count 字段.
        """
        payload = event.get("data", {})
        timestamp = payload.get("localSec", event.get("ts", 0))
        count = payload.get("count", 0)

        self._last_event_ts = max(self._last_event_ts, timestamp)
        self._current_people_count = count

        # 当人员少于 1 人时，开始计时，持续 10S 才进行一次记录
        if count < 1:
            if self._no_people_start_ts is None:
                self._no_people_start_ts = timestamp
                logger.info(
                    f"人员状态监控: 开始记录无人值守状态计时 @{timestamp:.1f}s"
                )
        else:
            # 人数恢复到 >= 1 人，检查是否满足 10S 阈值
            duration_sec = timestamp - self._no_people_start_ts
            if duration_sec >= self._no_people_sec:
                description = (
                    f"警报：主控室无人值守（人员少于1人）持续超过"
                    f"{self._no_people_sec:g}S（实际持续 {duration_sec:.1f}秒）"
                )
                logger.warning(
                    f"人员状态违规 (PeopleCount): {description} @{timestamp:.1f}s"
                )

                violation = {
                    "localSec": self._no_people_start_ts,
                    "event": "NO_OPERATOR_VIOLATION",
                    "count": 0,
                    "duration": round(duration_sec, 2),
                    "source": "tracker",
                    "description": description,
                }
                self._violations.append(violation)
            else:
                logger.info(
                    f"人员状态监控: 无人值守时长仅 {duration_sec:.1f}秒，"
                    f"未达 {self._no_people_sec:g}S 阈值，不作记录"
                )
            self._no_people_start_ts = None

    def _on_identity_away(self, event: Dict[str, Any]) -> None:
        """处理身份模块发布的工位离岗事件（IDENTITY_AWAY）.

        Args:
            event: 事件载荷；data 含 identity / event(away_start|away_end) /
                duration_sec / localSec.
        """
        payload = event.get("data", {})
        identity = payload.get("identity") or ""
        event_type = payload.get("event")
        timestamp = float(payload.get("localSec", event.get("ts", 0)) or 0.0)
        self._last_event_ts = max(self._last_event_ts, timestamp)

        if event_type == "away_start":
            # localSec 即离岗起点：identity 已按 away_hold_sec 防抖确认，
            # 故此处直接记违规，不再等第二道时长阈值
            cn = _IDENTITY_CN.get(identity, identity)
            violation = {
                "localSec": round(timestamp, 2),
                "event": "STATION_AWAY_VIOLATION",
                "identity": identity,
                "duration": None,
                "source": "identity",
                "description": f"警报：{cn}离岗",
            }
            logger.warning(
                f"人员状态违规 (StationAway): {cn}离岗 @{timestamp:.1f}s"
            )
            self._violations.append(violation)
            self._away_start[identity] = timestamp
            self._open_violation[identity] = violation
            return

        if event_type != "away_end":
            return

        start = self._away_start.pop(identity, None)
        violation = self._open_violation.pop(identity, None)
        duration = payload.get("duration_sec")
        if duration is None and start is not None:
            duration = timestamp - start
        if violation is None or duration is None:
            return
        cn = _IDENTITY_CN.get(identity, identity)
        violation["duration"] = round(float(duration), 2)
        violation["description"] = (
            f"警报：{cn}离岗，已持续 {float(duration):.1f}秒"
        )
        logger.info(
            f"人员状态监控: {cn}返岗（离岗 {float(duration):.1f}秒）"
        )

    def save_results(self, result_dir: str) -> None:
        """把违规明细落盘到 result_dir/rules/personnel_status_violations.json.

        Args:
            result_dir: 结果目录路径；为空时退回 set_result_dir 设置的目录.
        """
        target_dir = result_dir or self._result_dir
        if not target_dir:
            logger.warning("结果目录未设置，跳过 personnel_status 违规落盘")
            return
        try:
            rules_dir = os.path.join(target_dir, "rules")
            os.makedirs(rules_dir, exist_ok=True)
            path = os.path.join(rules_dir, VIOLATIONS_FILENAME)
            payload = {
                "source": "rules.personnel_status",
                "count": len(self._violations),
                "thresholds": {
                    "gaze_away_sec": self._gaze_away_sec,
                    "no_people_sec": self._no_people_sec,
                    # 离岗口径沿用 identity 模块，不在本制度内部另设阈值
                    "station_away_source": "identity.away_hold_sec",
                },
                "violations": self._violations,
            }
            with open(path, "w", encoding="utf-8") as file_obj:
                json.dump(payload, file_obj, ensure_ascii=False, indent=2)
            logger.info(f"保存 {len(self._violations)} 条违规到 {path}")
        except Exception as error:
            logger.error(f"保存人员状态违规失败: {error}", exc_info=True)

    def reset(self) -> None:
        """重置人员状态监控，防止新一轮推理混入旧的违规记录."""
        self.finalize()
        self._violations = []
        self._current_people_count = -1
        self._no_people_start_ts = None
        self._last_event_ts = 0.0
        self._away_start = {}
        self._open_violation = {}


def register(config: Optional[Dict[str, Any]] = None) -> PersonnelStatusRule:
    """实例化并返回人员状态监控制度实例.

    Args:
        config: `rules.params.personnel_status` 配置段（阈值为其键），可为 None.

    Returns:
        可注册到注册表的人员状态监控制度实例.
    """
    return PersonnelStatusRule(config)
