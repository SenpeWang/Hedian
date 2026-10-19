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

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# 机箱(ROI)区域定义（唯一真源，相对 camPOP 1920x1080）
SCREEN_POLYGONS: List[List[Tuple[int, int]]] = [
    [
        (1715, 473),
        (1737, 473),
        (1680, 802),
        (1578, 715),
        (1543, 539),
    ],  # case_A
    [(1070, 1014), (1592, 699), (1692, 786), (1216, 1081)],  # case_B
    # case_C_visible
    [(884, 1081), (1025, 982), (1113, 1081)],
]

SCREEN_POLYGON_LABELS: List[str] = ["case_A", "case_B", "case_C_visible"]


def screen_polygons_as_tuples() -> List[Tuple[str, np.ndarray]]:
    """把模块级 ROI 定义转成 (label, contour) 结构，供 draw_roi_overlay 使用.

    Returns:
        与 SCREEN_POLYGONS 顺序对应的 (标签, 顶点数组) 元组列表，
        顶点数组为 int32 类型.
    """
    return [
        (label, np.array(polygon, dtype=np.int32))
        for label, polygon in zip(SCREEN_POLYGON_LABELS, SCREEN_POLYGONS)
    ]


# 机箱多边形标注基准分辨率（实际视频分辨率不同时按比例缩放）
BASE_WIDTH, BASE_HEIGHT = 1920, 1080


def scale_screen_polygons(
    width: int, height: int
) -> List[Tuple[str, np.ndarray]]:
    """把机箱多边形按实际帧尺寸等比缩放，供推理与绘制共用.

    Args:
        width: 实际帧宽.
        height: 实际帧高.

    Returns:
        与 SCREEN_POLYGONS 顺序对应的 (标签, 缩放后轮廓) 列表，
        轮廓为 int32 类型.
    """
    scale_x, scale_y = width / BASE_WIDTH, height / BASE_HEIGHT
    if abs(scale_x - 1.0) < 1e-3 and abs(scale_y - 1.0) < 1e-3:
        return screen_polygons_as_tuples()
    return [
        (
            label,
            np.array(
                [
                    (int(round(x * scale_x)), int(round(y * scale_y)))
                    for x, y in polygon
                ],
                dtype=np.int32,
            ),
        )
        for label, polygon in zip(SCREEN_POLYGON_LABELS, SCREEN_POLYGONS)
    ]


# 可视化绘制
def draw_roi_overlay(
    image: np.ndarray,
    polygons: List[Tuple[str, np.ndarray]],
    color: Tuple[int, int, int] = (0, 200, 255),
    alpha: float = 0.35,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """绘制半透明 ROI 多边形（实心边框 + 质心标签）.

    Args:
        image: 原始视频帧 (BGR 格式)，原地绘制.
        polygons: (标签, 多边形顶点数组) 元组列表.
        color: BGR 颜色，默认统一黄色 (0, 200, 255).
        alpha: 半透明填充系数.
        font_scale: 标签字号.
        thickness: 边框线宽.

    Returns:
        完成绘制后的同一帧数组，与传入的 image 为同一对象.
    """
    overlay = image.copy()

    for label, contour in polygons:
        vertices = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
        flat_vertices = vertices.reshape(-1, 2)
        cv2.fillPoly(overlay, [flat_vertices], color)
        cv2.polylines(image, [flat_vertices], True, color, thickness)
        centroid = flat_vertices.mean(axis=0).astype(int)
        cv2.putText(
            image,
            label,
            tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


def draw_detection_boxes(
    frame: np.ndarray,
    hands: Optional[List[Any]] = None,
    files: Optional[List[Any]] = None,
) -> None:
    """绘制检测到的目标框：pointing_hand 紫框 / file 黄框.

    Args:
        frame: 视频帧，原地绘制.
        hands: pointing_hand 检测框列表.
        files: file 检测框列表.
    """
    for bbox in files or []:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            frame,
            "file",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )
    for bbox in hands or []:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)


def draw_trigger(
    frame: np.ndarray,
    hands: Optional[List[Any]],
    label: str,
) -> None:
    """绘制事件触发框（红色粗框 + 触发标签）.

    Args:
        frame: 视频帧，原地绘制.
        hands: 触发事件的目标框列表.
        label: 触发标签文本.
    """
    for bbox in hands or []:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )


# MediaPipe 21 关键点手部骨架连接（0 手腕, 1-4 拇指, 5-8 食指, 9-12 中指,
# 13-16 无名指, 17-20 小指）
HAND_EDGES: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

# 检测框配色（BGR）: hand 橙 / file 黄 / person 绿 / pointing_hand 红
CLASS_BOX_COLORS: Dict[str, Tuple[int, int, int]] = {
    "hand": (0, 200, 255),
    "file": (255, 255, 0),
    "person": (0, 255, 0),
    "pointing_hand": (0, 0, 255),
}


def draw_class_boxes(
    frame: np.ndarray,
    detections: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """按类别配色绘制 YOLO 检测框与类名标签.

    Args:
        frame: 视频帧，原地绘制.
        detections: 检测项列表，每项含 bbox / name / confidence 键.
    """
    for detection in detections or []:
        bbox = detection["bbox"]
        name = str(detection.get("name", ""))
        confidence = float(detection.get("confidence", 0.0))
        color = CLASS_BOX_COLORS.get(name, (200, 200, 200))
        x1, y1, x2, y2 = (int(value) for value in bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )


def draw_hand_landmarks(
    frame: np.ndarray,
    hands: Optional[List[Dict[str, Any]]] = None,
    draw_all_hands: bool = True,
) -> None:
    """绘制手部 21 关键点骨架：指向手 = 红框 + 白线，其余手 = 橙框 + 灰线.

    Args:
        frame: 视频帧，原地绘制.
        hands: 手部结果列表，每项含 bbox / landmarks / pointing 键.
        draw_all_hands: 是否绘制所有检出到关键点的手；False 时只画指向手.
    """
    for hand in hands or []:
        landmarks = hand.get("landmarks")
        if landmarks is None:
            continue
        is_pointing = bool(hand.get("pointing"))
        if not draw_all_hands and not is_pointing:
            continue

        box_color = (0, 0, 255) if is_pointing else (0, 200, 255)
        skeleton_color = (255, 255, 255) if is_pointing else (160, 160, 160)
        x1, y1, x2, y2 = (int(value) for value in hand["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        for start_index, end_index in HAND_EDGES:
            start_point = (
                int(landmarks[start_index][0]),
                int(landmarks[start_index][1]),
            )
            end_point = (
                int(landmarks[end_index][0]),
                int(landmarks[end_index][1]),
            )
            cv2.line(frame, start_point, end_point, skeleton_color, 2)
        for point in landmarks:
            cv2.circle(
                frame,
                (int(point[0]), int(point[1])),
                3,
                skeleton_color,
                -1,
            )
