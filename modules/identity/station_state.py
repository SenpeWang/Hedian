"""工位在岗/离岗状态机（纯逻辑，无 IO、无第三方依赖，便于单测）.

判定口径：以人（身份）为单位，身体中心是否落在该身份的工位多边形内。
连续离岗超过 away_hold_sec 才确认"离岗开始"，回来即刻确认"离岗结束",
避免侧身/短暂遮挡造成大量碎段。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("module.identity.station")

Point = Tuple[float, float]
Polygon = Sequence[Point]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """射线法判断点是否在多边形内（含边界近似）.

    Args:
        point: 待判定点 (x, y)，像素坐标.
        polygon: 多边形顶点序列 (x, y)，首尾不必闭合.

    Returns:
        点在多边形内返回 True.
    """
    if not polygon or len(polygon) < 3:
        return False
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_cross:
                inside = not inside
    return inside


class StationMonitor:
    """按身份维护在岗/离岗状态，产出离岗事件.

    事件字典结构：
        {"type": "away_start"|"away_end", "identity": str,
         "start": float, "end": Optional[float], "duration": Optional[float]}
    """

    def __init__(
        self,
        polygons_px: Dict[str, Polygon],
        away_hold_sec: float = 2.0,
    ) -> None:
        """初始化.

        Args:
            polygons_px: 身份 -> 工位像素多边形.
            away_hold_sec: 连续离岗超过该秒数才确认离岗（防抖）.
        """
        self._polygons: Dict[str, Polygon] = dict(polygons_px or {})
        self._away_hold = max(0.0, float(away_hold_sec))
        # identity -> {"absent_since": float|None, "away_start": float|None}
        self._state: Dict[str, Dict[str, Optional[float]]] = {
            identity: {"absent_since": None, "away_start": None}
            for identity in self._polygons
        }

    @property
    def identities(self) -> List[str]:
        """已配置工位多边形的身份列表."""
        return list(self._polygons)

    def is_on_station(self, identity: str, center: Optional[Point]) -> bool:
        """判断某身份当前是否在其工位多边形内.

        Args:
            identity: 'LEADER' | 'ROAD1' | 'ROAD2'.
            center: 人体中心像素坐标；None 表示本帧没跟到该身份.

        Returns:
            在工位内返回 True.
        """
        if center is None:
            return False
        return point_in_polygon(center, self._polygons.get(identity) or [])

    def is_away(self, identity: str) -> bool:
        """该身份当前是否已确认离岗（已超过防抖阈值）.

        Args:
            identity: 身份名.

        Returns:
            已确认离岗返回 True；仅在防抖期内或未离岗返回 False.
        """
        state = self._state.get(identity)
        return bool(state and state.get("away_start") is not None)

    def away_seconds(self, identity: str, timestamp: float) -> float:
        """返回该身份本次连续离岗已持续秒数（未离岗为 0）.

        Args:
            identity: 身份名.
            timestamp: 当前视频时间（秒）.

        Returns:
            连续离岗秒数.
        """
        state = self._state.get(identity)
        if not state or state.get("absent_since") is None:
            return 0.0
        return max(0.0, timestamp - float(state["absent_since"]))

    def update(
        self,
        timestamp: float,
        identity_centers: Dict[str, Optional[Point]],
        active: Optional[set] = None,
    ) -> List[dict]:
        """用本帧各身份的位置更新状态机.

        Args:
            timestamp: 当前视频时间（秒）.
            identity_centers: 身份 -> 中心点（None 表示本帧未跟踪到）.
            active: 当前已锁定身份的集合；given 时未锁定的身份不参与判定
                （避免"身份还没锁上"被误判成离岗）。角色被释放时，
                若存在未闭合的离岗段，就地闭合。

        Returns:
            本帧新产生的状态事件列表（away_start / away_end）.
        """
        events: List[dict] = []
        for identity in self._polygons:
            state = self._state[identity]
            if active is not None and identity not in active:
                away_start = state.get("away_start")
                if away_start is not None:
                    events.append(
                        {
                            "type": "away_end",
                            "identity": identity,
                            "start": float(away_start),
                            "end": float(timestamp),
                            "duration": round(float(timestamp) - float(away_start), 2),
                        }
                    )
                state["absent_since"] = None
                state["away_start"] = None
                continue
            on_station = self.is_on_station(identity, identity_centers.get(identity))

            if on_station:
                away_start = state.get("away_start")
                if away_start is not None:
                    events.append(
                        {
                            "type": "away_end",
                            "identity": identity,
                            "start": float(away_start),
                            "end": float(timestamp),
                            "duration": round(float(timestamp) - float(away_start), 2),
                        }
                    )
                state["absent_since"] = None
                state["away_start"] = None
                continue

            # 不在工位：先记录离开起点，超过防抖阈值才确认离岗
            if state.get("absent_since") is None:
                state["absent_since"] = float(timestamp)
            if (
                state.get("away_start") is None
                and float(timestamp) - float(state["absent_since"]) >= self._away_hold
            ):
                state["away_start"] = float(state["absent_since"])
                events.append(
                    {
                        "type": "away_start",
                        "identity": identity,
                        "start": float(state["away_start"]),
                        "end": None,
                        "duration": None,
                    }
                )
                logger.info(
                    "确认离岗 %s (自 %.1fs 起, 已持续 %.1fs)",
                    identity,
                    float(state["away_start"]),
                    float(timestamp) - float(state["away_start"]),
                )
        return events

    def close(self, timestamp: float) -> List[dict]:
        """视频结束时收尾：把未闭合的离岗段就地闭合.

        Args:
            timestamp: 末帧视频时间（秒）.

        Returns:
            补发的 away_end 事件列表.
        """
        events: List[dict] = []
        for identity, state in self._state.items():
            away_start = state.get("away_start")
            if away_start is None:
                continue
            events.append(
                {
                    "type": "away_end",
                    "identity": identity,
                    "start": float(away_start),
                    "end": float(timestamp),
                    "duration": round(float(timestamp) - float(away_start), 2),
                }
            )
            state["away_start"] = None
            state["absent_since"] = None
        return events
