"""行为检测通用工具（几何纯函数 + 姿态工具）.

本模块集中存放 behavior 各检测器共用的纯函数与工具类：
  - 几何纯函数：点/线段/多边形距离、多边形重叠比（移植自 behavior-v1/finger_screen_pop.py）
  - 统一 IoU：bbox_iou（合并原 file_detector._iou 与 hand_raiser._bbox_iou）
  - 姿态工具：PoseTools（夹角计算）、PoseEMAFilter（关键点 EMA 平滑）

所有函数均为无副作用纯函数，供 screen_detect / file_detector / hand_raiser /
object_detector 等模块复用。
"""
from typing import Dict, Optional, Tuple

import numpy as np


# 几何工具（移植自 behavior-v1/finger_screen_pop.py，模块级纯函数）
def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """射线法判断点是否在多边形内.

    Args:
        point: 待判断的点坐标 (x, y).
        polygon: 多边形顶点数组，形状 (n, 2).

    Returns:
        点位于多边形内返回 True，否则返回 False.
    """
    point_x, point_y = point
    vertex_count = len(polygon)
    inside = False
    previous_index = vertex_count - 1
    for current_index in range(vertex_count):
        current_x, current_y = polygon[current_index]
        previous_x, previous_y = polygon[previous_index]
        if ((current_y > point_y) != (previous_y > point_y)) and (
            point_x < (previous_x - current_x) * (point_y - current_y) /
            (previous_y - current_y + 1e-9) + current_x
        ):
            inside = not inside
        previous_index = current_index
    return inside


def point_to_segment_distance(
    point: Tuple[float, float],
    segment_start: Tuple[float, float],
    segment_end: Tuple[float, float],
) -> float:
    """点到线段的最短距离，投影落在端点外侧时取端点距离.

    Args:
        point: 待判断的点坐标 (x, y).
        segment_start: 线段起点坐标.
        segment_end: 线段终点坐标.

    Returns:
        点到线段的最短欧氏距离；线段退化（长度接近 0）时取到起点的距离.
    """
    point_x, point_y = point
    start_x, start_y = segment_start
    end_x, end_y = segment_end
    edge_x, edge_y = end_x - start_x, end_y - start_y
    offset_x, offset_y = point_x - start_x, point_y - start_y
    denominator = edge_x * edge_x + edge_y * edge_y
    if denominator < 1e-9:
        return float(np.hypot(offset_x, offset_y))
    parameter = max(
        0.0, min(1.0, (offset_x * edge_x + offset_y * edge_y) / denominator)
    )
    closest_x = start_x + parameter * edge_x
    closest_y = start_y + parameter * edge_y
    return float(np.hypot(point_x - closest_x, point_y - closest_y))


def point_to_polygon_distance(
    point: Tuple[float, float], polygon: np.ndarray,
) -> float:
    """点到多边形轮廓的最短距离，点位于多边形内时返回 0.

    Args:
        point: 待判断的点坐标 (x, y).
        polygon: 多边形顶点数组，形状 (n, 2).

    Returns:
        点到多边形各边的最短距离最小值；点在多边形内返回 0.
    """
    if point_in_polygon(point, polygon):
        return 0.0
    vertices = polygon.reshape(-1, 2)
    return min(
        point_to_segment_distance(
            point, tuple(vertices[index]), tuple(vertices[(index + 1) % len(vertices)])
        )
        for index in range(len(vertices)))


def bbox_to_polygon_distance(bbox, polygon: np.ndarray) -> float:
    """检测框四角到多边形的最短距离.

    Args:
        bbox: 检测框 [x1, y1, x2, y2].
        polygon: 多边形顶点数组，形状 (n, 2).

    Returns:
        检测框四个角点到多边形的最短距离最小值.
    """
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return min(point_to_polygon_distance(corner, polygon) for corner in corners)


def bbox_polygon_overlap_ratio(bbox, polygon: np.ndarray) -> float:
    """检测框与多边形的重叠比，即交集面积除以检测框面积.

    Args:
        bbox: 检测框 [x1, y1, x2, y2].
        polygon: 多边形顶点数组，形状 (n, 2).

    Returns:
        检测框与多边形交集面积占检测框面积的比例，取值 0~1.
    """
    x1, y1, x2, y2 = bbox
    # 采样检测框内部网格点判断是否在多边形内
    grid_size_x, grid_size_y = 8, 8
    inside_count = 0
    total_count = 0
    for column_index in range(grid_size_x):
        for row_index in range(grid_size_y):
            sample_x = x1 + (column_index + 0.5) / grid_size_x * (x2 - x1)
            sample_y = y1 + (row_index + 0.5) / grid_size_y * (y2 - y1)
            total_count += 1
            if point_in_polygon((sample_x, sample_y), polygon):
                inside_count += 1
    return inside_count / max(1, total_count)


def bbox_iou(box_a, box_b) -> float:
    """计算两个 bbox [x1, y1, x2, y2] 的 IoU.

    统一合并原 file_detector._iou 与 hand_raiser._bbox_iou，
    采用 `union > 0` 边界保护，避免除零。

    Args:
        box_a: 检测框 A 的 [x1, y1, x2, y2].
        box_b: 检测框 B 的 [x1, y1, x2, y2].

    Returns:
        两个检测框的交并比，两框面积之和为 0 时返回 0.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
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
    def calc_angle(
        point_a: np.ndarray,
        point_b: np.ndarray,
        point_c: np.ndarray,
    ) -> Optional[float]:
        """计算三点构成的角度（度），点 b 为顶点，退化时返回 None.

        Args:
            point_a: 端点 A 坐标及置信度 [x, y, conf]，或 [x, y].
            point_b: 顶点 B 坐标及置信度 [x, y, conf]，或 [x, y].
            point_c: 端点 C 坐标及置信度 [x, y, conf]，或 [x, y].

        Returns:
            角度（度）；两向量任一长度退化时返回 None.
        """
        vector_ba = np.array(
            [point_a[0] - point_b[0], point_a[1] - point_b[1]], dtype=float
        )
        vector_bc = np.array(
            [point_c[0] - point_b[0], point_c[1] - point_b[1]], dtype=float
        )
        norm_ba = np.linalg.norm(vector_ba)
        norm_bc = np.linalg.norm(vector_bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return None
        cos_angle = float(np.clip(
            np.dot(vector_ba, vector_bc) / (norm_ba * norm_bc), -1.0, 1.0
        ))
        return float(np.degrees(np.arccos(cos_angle)))


class PoseEMAFilter:
    """关键点 EMA 平滑滤波器（对齐 behavior-v1）.

    对每个跟踪目标的关键点做指数移动平均，抑制单帧抖动；
    低置信度关键点（confidence < conf_threshold）沿用历史值，避免噪声误判。
    """

    def __init__(self, alpha: float = 0.5, conf_threshold: float = 0.25) -> None:
        """初始化 EMA 滤波器.

        Args:
            alpha: EMA 平滑系数，取 0~1 之间.
            conf_threshold: 关键点有效置信度阈值.
        """
        self.alpha: float = alpha
        self.conf_threshold: float = conf_threshold
        self.history: Dict[object, np.ndarray] = {}

    def update(self, track_id: int, keypoints: np.ndarray) -> np.ndarray:
        """更新并返回平滑后的关键点.

        Args:
            track_id: 外部跟踪标识，作为历史关键点缓存的键.
            keypoints: 当前帧关键点数组，形状 (17, 3)，末列为置信度.

        Returns:
            平滑后的关键点数组；首次更新或低置信度时沿用历史值.
        """
        if track_id not in self.history:
            self.history[track_id] = keypoints.copy()
            return keypoints
        previous_keypoints = self.history[track_id]
        smoothed = np.zeros_like(keypoints)
        for keypoint_index in range(len(keypoints)):
            x, y, confidence = keypoints[keypoint_index]
            previous_x, previous_y, previous_confidence = (
                previous_keypoints[keypoint_index]
            )
            if confidence < self.conf_threshold:
                smoothed[keypoint_index] = [
                    previous_x, previous_y, previous_confidence,
                ]
            else:
                smoothed[keypoint_index] = [
                    self.alpha * x + (1.0 - self.alpha) * previous_x,
                    self.alpha * y + (1.0 - self.alpha) * previous_y,
                    confidence,
                ]
        self.history[track_id] = smoothed.copy()
        return smoothed
