"""行为检测通用工具（几何纯函数 + 姿态工具）.

本模块集中存放 behavior 各检测器共用的纯函数与工具类：
  - 几何纯函数：点/线段/多边形距离、多边形重叠比（移植自 behavior-v1/finger_screen_pop.py）
  - 统一 IoU：bbox_iou（合并原 file_detector._iou 与 hand_raiser._bbox_iou）
  - 姿态工具：PoseTools（夹角计算）、PoseEMAFilter（关键点 EMA 平滑）

所有函数均为无副作用纯函数，供 screen_detect / file_detector / hand_raiser /
object_detector 等模块复用。
"""
from typing import Optional, Tuple

import numpy as np


# 几何工具（移植自 behavior-v1/finger_screen_pop.py，模块级纯函数）
def point_in_polygon(pt: Tuple[float, float], poly: np.ndarray) -> bool:
    """射线法判断点是否在多边形内."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) /
                                       (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def point_to_segment_dist(p, a, b) -> float:
    """点 p 到线段 ab 的最短距离."""
    px, py = p
    ax, ay = a
    bx, by = b
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom < 1e-9:
        return float(np.hypot(apx, apy))
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx, cy = ax + t * abx, ay + t * aby
    return float(np.hypot(px - cx, py - cy))


def point_to_polygon_dist(p, poly: np.ndarray) -> float:
    """点 p 到多边形轮廓的最短距离（含内部返回 0）."""
    if point_in_polygon(p, poly):
        return 0.0
    pts = poly.reshape(-1, 2)
    return min(
        point_to_segment_dist(p, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]))
        for i in range(len(pts)))


def bbox_to_polygon_dist(bbox, poly: np.ndarray) -> float:
    """Bbox 四角到多边形的最短距离."""
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return min(point_to_polygon_dist(c, poly) for c in corners)


def bbox_polygon_overlap_ratio(bbox, poly: np.ndarray) -> float:
    """Bbox 与多边形的重叠比（交集面积 / bbox 面积）."""
    x1, y1, x2, y2 = bbox
    # 采样 bbox 内部网格点判断是否在多边形内
    nx, ny = 8, 8
    inside = 0
    total = 0
    for i in range(nx):
        for j in range(ny):
            px = x1 + (i + 0.5) / nx * (x2 - x1)
            py = y1 + (j + 0.5) / ny * (y2 - y1)
            total += 1
            if point_in_polygon((px, py), poly):
                inside += 1
    return inside / max(1, total)


def bbox_iou(a, b) -> float:
    """计算两个 bbox [x1, y1, x2, y2] 的 IoU.

    统一合并原 file_detector._iou 与 hand_raiser._bbox_iou，
    采用 `union > 0` 边界保护，避免除零。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# 姿态工具
class PoseTools:
    """姿态几何工具."""

    @staticmethod
    def calc_angle(a, b, c) -> Optional[float]:
        """
        计算 ∠abc（顶点 b）的角度（度）.

        a, b, c: [x, y, conf] 或 [x, y]
        """
        ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
        bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
        nba = np.linalg.norm(ba)
        nbc = np.linalg.norm(bc)
        if nba < 1e-6 or nbc < 1e-6:
            return None
        cos_a = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_a)))


class PoseEMAFilter:
    """关键点 EMA 平滑滤波器（对齐 behavior-v1）.

    对每个跟踪目标的关键点做指数移动平均，抑制单帧抖动；
    低置信度关键点（conf < conf_thres）沿用历史值，避免噪声误判。
    """

    def __init__(self, alpha: float = 0.5, conf_thres: float = 0.25):
        """初始化."""
        self.alpha = alpha
        self.conf_thres = conf_thres
        self.history: dict = {}

    def update(self, key, keypoints: np.ndarray) -> np.ndarray:
        """更新并返回平滑后的关键点（key 为外部 track_id）."""
        if key not in self.history:
            self.history[key] = keypoints.copy()
            return keypoints
        prev = self.history[key]
        smoothed = np.zeros_like(keypoints)
        for i in range(len(keypoints)):
            cx, cy, cc = keypoints[i]
            px, py, pc = prev[i]
            if cc < self.conf_thres:
                smoothed[i] = [px, py, pc]  # 低置信度沿用历史
            else:
                smoothed[i] = [
                    self.alpha * cx + (1.0 - self.alpha) * px,
                    self.alpha * cy + (1.0 - self.alpha) * py,
                    cc,
                ]
        self.history[key] = smoothed.copy()
        return smoothed
