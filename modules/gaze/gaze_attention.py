"""注视关注度判定模块.

在给定时间窗口内基于注视点累计位移判断视线是否发生转动：累计位移达到
阈值视为视线转动，即认为给予了关注；未达到则视为未给予关注。
"""
import math
from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GazePoint:
    """单个注视点记录.

    Attributes:
        timestamp_ms: 记录时刻（毫秒）。
        x: 注视点 x 坐标（像素）。
        y: 注视点 y 坐标（像素）。
    """

    timestamp_ms: float
    x: float
    y: float


@dataclass
class AttentionResult:
    """注视关注度判定结果.

    Attributes:
        has_turned: 视线是否转动。True 表示视线转动即给予了关注，
            False 表示视线未转动即未给予关注。
    """

    has_turned: bool


# ---------------------------------------------------------------------------
# GazeAttentionChecker
# ---------------------------------------------------------------------------


class GazeAttentionChecker:
    """注视关注度检查器.

    在给定的时间窗口内分析注视点序列，判断视线是否发生转动：累计位移达到
    阈值即视为视线转动，也就是给予了关注。阈值为可配置的经验值，需根据
    真实视频数据校准。
    """

    def __init__(
        self,
        min_turn_displacement: float = 100.0,
        min_samples: int = 5,
    ) -> None:
        """初始化关注度检查器.

        Args:
            min_turn_displacement: 转动判定下限（像素），累计位移达到此值视为转动。
            min_samples: 判定所需的最少样本数。
        """
        self._min_turn_displacement = min_turn_displacement
        self._min_samples = min_samples

    def evaluate(
        self,
        gaze_points: List[GazePoint],
    ) -> AttentionResult:
        """评估一组注视点是否发生转动，即是否给予了关注.

        Args:
            gaze_points: 按时间排序的注视点列表。

        Returns:
            AttentionResult；has_turned 为 True 表示视线转动即给予关注。
        """
        point_count = len(gaze_points)
        if point_count < self._min_samples:
            return AttentionResult(has_turned=False)

        # 累计窗口内的帧间欧氏位移
        total_displacement = 0.0
        for index in range(1, point_count):
            dx = gaze_points[index].x - gaze_points[index - 1].x
            dy = gaze_points[index].y - gaze_points[index - 1].y
            total_displacement += math.sqrt(dx * dx + dy * dy)

        # 位移达到阈值即视为转动，也就是给予关注
        return AttentionResult(
            has_turned=total_displacement >= self._min_turn_displacement
        )
