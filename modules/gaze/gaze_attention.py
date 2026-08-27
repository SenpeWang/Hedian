"""注视关注度判定模块.

在给定时间窗口内，基于注视点累计位移判断视线是否发生了转动：
- 累计位移 >= 阈值 → 视线转动 → 认为给予了关注
- 累计位移 <  阈值 → 视线未转动 → 认为未给予关注
"""

import math
from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GazePoint:
    """单个注视点记录."""

    timestamp_ms: float
    gx: float    # 注视点 x 坐标（像素）
    gy: float    # 注视点 y 坐标（像素）


@dataclass
class AttentionResult:
    """注视关注度判定结果."""

    has_turned: bool              # True=视线转动（给予了关注）, False=未转动（未关注）


# ---------------------------------------------------------------------------
# GazeAttentionChecker
# ---------------------------------------------------------------------------


class GazeAttentionChecker:
    """注视关注度检查器.

    在给定的时间窗口内，分析注视点序列，判断视线是否发生了转动。
    判定逻辑：累计位移 >= 阈值 → 视线转动 → 给予了关注。
    阈值为可配置的经验值，需根据真实视频数据校准。
    """

    def __init__(
        self,
        min_turn_displacement: float = 100.0,
        min_samples: int = 5,
    ):
        """
        初始化注意力估算器.

        Args:
            min_turn_displacement: 转动判定下限（像素）。累计位移达到此值视为转动。
            min_samples: 最少需要的样本数。
        """
        self._min_turn_displacement = min_turn_displacement
        self._min_samples = min_samples

    def evaluate(
        self,
        gaze_points: List[GazePoint],
    ) -> AttentionResult:
        """评估一组注视点是否发生了转动（即是否给予了关注）.

        Args:
            gaze_points: 按时间排序的注视点列表。

        Returns:
            AttentionResult，has_turned=True 表示视线转动=给予关注。
        """
        n = len(gaze_points)
        if n < self._min_samples:
            return AttentionResult(has_turned=False)

        # 计算窗口内累计帧间欧氏位移
        total_disp = 0.0
        for i in range(1, n):
            dx = gaze_points[i].gx - gaze_points[i - 1].gx
            dy = gaze_points[i].gy - gaze_points[i - 1].gy
            total_disp += math.sqrt(dx * dx + dy * dy)

        # 转动判定：位移 >= 阈值 → 转动了 → 给予关注
        return AttentionResult(
            has_turned=total_disp >= self._min_turn_displacement
        )
