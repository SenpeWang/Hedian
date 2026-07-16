"""
人员状态监控制度

实现 BaseRegulation 接口，管理人员状态监控规程。

包含两条规则：
1. 监控操纵员，确保 2 人中至少 1 人监控机组，当两名操纵员视线同时离开盘台超过 1 分钟时，进行一次记录。
2. 监视主控室人数，当人员少于 1 人时，进行一次记录。
"""
import logging
from typing import Dict, Optional, List

from core.event_bus import EventBus, EventTopic
from rules.rule_base import BaseRule

logger = logging.getLogger("rules.personnel_status")


class PersonnelStatusRule(BaseRule):
    """人员状态监控制度"""

    def __init__(self, config: dict = None):
        """
        初始化人员状态监控制度
        """
        self._config = config or {}
        self._event_bus = None
        
        # 记录所有的违规情况
        self._violations: List[dict] = []
        # 当前的主控室人数状态
        self._current_people_count = -1

        # 状态计时器与缓存时间戳
        self._no_people_start_ts = None
        self._last_ts = 0.0

    def name(self) -> str:
        """制度名称"""
        return "personnel_status"

    def subscribe_events(self, event_bus: EventBus) -> None:
        """订阅事件"""
        self._event_bus = event_bus
        event_bus.subscribe(EventTopic.GAZE_ALERT, self._on_gaze_alert)
        event_bus.subscribe(EventTopic.TRACKER_HEADCOUNT, self._on_people_count)

    def is_active(self) -> bool:
        """人员状态监控始终是激活状态"""
        return True

    def get_current_flow(self) -> Optional[dict]:
        """获取当前监控状态摘要"""
        return {
            "type": "personnel_status_monitoring",
            "current_people_count": self._current_people_count,
            "violations_count": len(self._violations),
        }

    def finalize(self) -> Optional[dict]:
        """视频结束时关闭"""
        # 视频收尾时，检查是否仍有未结束的无人值守状态且时长已超过 10 秒
        if self._no_people_start_ts is not None:
            duration = self._last_ts - self._no_people_start_ts
            if duration >= 10.0:
                description = f"警报：主控室无人值守（人员少于1人）持续超过10S（实际持续 {duration:.1f}秒）"
                logger.warning(f"人员状态违规 (PeopleCount/Finalize): {description} @{self._last_ts:.1f}s")
                violation = {
                    "localSec": self._no_people_start_ts,
                    "event": "NO_OPERATOR_VIOLATION",
                    "count": 0,
                    "duration": round(duration, 2),
                    "source": "tracker",
                    "description": description,
                }
                self._violations.append(violation)
            self._no_people_start_ts = None
        return None

    def _on_gaze_alert(self, msg: dict) -> None:
        """处理两名操纵员视线同时离开盘台超过 60 秒的事件"""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0))
        self._last_ts = max(self._last_ts, ts)
        away_duration = data.get("away_duration", 0)
        heads_count = data.get("heads_count", 0)

        # 只有持续超过60秒才记录违规
        if away_duration < 60:
            logger.debug(f"凝视告警忽略: 仅 {away_duration:.1f}秒，未达60秒阈值")
            return

        description = f"警报：两名操纵员视线同时离开盘台持续超过1分钟（实际持续 {away_duration:.1f}秒）"
        logger.warning(f"人员状态违规 (Gaze): {description} @{ts:.1f}s")

        # 记录违规
        violation = {
            "localSec": ts,
            "event": "PANEL_VIOLATION",
            "away_duration": away_duration,
            "heads_count": heads_count,
            "source": "gaze",
            "description": description,
        }
        self._violations.append(violation)

    def _on_people_count(self, msg: dict) -> None:
        """处理主控室人数变动事件"""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0))
        count = data.get("count", 0)

        self._last_ts = max(self._last_ts, ts)
        self._current_people_count = count

        # 当人员少于 1 人时，开始计时，持续 10S 才进行一次记录
        if count < 1:
            if self._no_people_start_ts is None:
                self._no_people_start_ts = ts
                logger.info(f"人员状态监控: 开始记录无人值守状态计时 @{ts:.1f}s")
        else:
            # 人数恢复到 >= 1 人，检查是否满足 10S 阈值
            if self._no_people_start_ts is not None:
                duration = ts - self._no_people_start_ts
                if duration >= 10.0:
                    description = f"警报：主控室无人值守（人员少于1人）持续超过10S（实际持续 {duration:.1f}秒）"
                    logger.warning(f"人员状态违规 (PeopleCount): {description} @{ts:.1f}s")
                    
                    violation = {
                        "localSec": self._no_people_start_ts,
                        "event": "NO_OPERATOR_VIOLATION",
                        "count": 0,
                        "duration": round(duration, 2),
                        "source": "tracker",
                        "description": description,
                    }
                    self._violations.append(violation)
                else:
                    logger.info(f"人员状态监控: 无人值守时长仅 {duration:.1f}秒，未达10S阈值，不作记录")
                self._no_people_start_ts = None

    def get_violations(self) -> list:
        """获取所有记录的违规"""
        return list(self._violations)


def register():
    """模块注册入口"""
    return PersonnelStatusRule()
