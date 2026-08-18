"""行为检测可视化（机箱 ROI 定义与检测结果绘制）— 单一真源.

包含两部分：
  1. 机箱多边形坐标定义（供推理与可视化共用，相对 camPOP 1920x1080）
  2. 可视化绘制函数：
     - draw_roi_overlay：机箱 ROI 半透明填充 + 边框 + 质心标签
     - draw_detection_boxes：检测到的目标框（pointing_hand 紫框 / file 黄框）
     - draw_trigger：事件触发框（红色粗框 + 触发标签）

检测器（screen_detect / file_detector）调用本模块绘制函数，不自行写 cv2 绘制代码。

可视化风格：统一黄色半透明填充 + 实线闭合边框 + 质心标签（对齐 gaze_roi_app/visualizer）。
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple

# 机箱(ROI)区域定义（唯一真源，相对 camPOP 1920x1080）
SCREEN_POLYGONS: List[List[Tuple[int, int]]] = [
    [(1715, 473), (1737, 473), (1680, 802), (1578, 715),
     (1543, 539)],  # case_A
    [(1070, 1014), (1592, 699), (1692, 786), (1216, 1081)],  # case_B
    # case_C_visible
    [(884, 1081), (1025, 982), (1113, 1081)],
]

SCREEN_POLYGON_LABELS: List[str] = ["case_A", "case_B", "case_C_visible"]


def screen_polygons_as_tuples() -> List[Tuple[str, np.ndarray]]:
    """转成 (label, contour) 结构，供 draw_roi_overlay 使用."""
    return [(lbl, np.array(poly, dtype=np.int32))
            for lbl, poly in zip(SCREEN_POLYGON_LABELS, SCREEN_POLYGONS)]


# 可视化绘制
def draw_roi_overlay(
    image: np.ndarray,
    polygons: List[Tuple[str, np.ndarray]],
    color: Tuple[int, int, int] = (0, 200, 255),  # 统一黄色 BGR
    alpha: float = 0.35,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """绘制半透明 ROI 多边形（实心边框 + 质心标签）."""
    overlay = image.copy()

    for label, contour in polygons:
        pts = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
        flat = pts.reshape(-1, 2)
        cv2.fillPoly(overlay, [flat], color)
        cv2.polylines(image, [flat], True, color, thickness)
        centroid = flat.mean(axis=0).astype(int)
        cv2.putText(image, label, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


def draw_detection_boxes(
    frame: np.ndarray,
    hands: Optional[List] = None,
    files: Optional[List] = None,
) -> None:
    """绘制检测到的目标框：pointing_hand 紫框 / file 黄框."""
    for box in files or []:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, "file", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1)
    for box in hands or []:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)


def draw_trigger(
    frame: np.ndarray,
    hands: Optional[List],
    label: str,
) -> None:
    """绘制事件触发框（红色粗框 + 触发标签）."""
    for box in hands or []:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
