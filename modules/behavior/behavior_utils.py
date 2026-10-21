"""行为检测通用工具（几何纯函数 + 指向手势 + 姿态工具）.

本模块集中存放 behavior 各检测器共用的纯函数与工具类：
  - 几何纯函数：点/线段/多边形距离、多边形重叠比（移植自 behavior-v1/finger_screen_pop.py）
  - 统一 IoU / 包含比：bbox_iou、bbox_containment_ratio
  - 指向手势：angle_at、finger_angles、judge_pointing、HandLandmarkExtractor
  - 姿态工具：PoseTools（夹角计算）、PoseEMAFilter（关键点 EMA 平滑）、
    PoseRoleClassifier（事件时刻判别机组长 / 操作员）

所有几何函数均为无副作用纯函数，供 screen_detect / file_detector / hand_raiser /
behavior_module 等模块复用。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("module.behavior.utils")


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
            point_x
            < (previous_x - current_x)
            * (point_y - current_y)
            / (previous_y - current_y + 1e-9)
            + current_x
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
    point: Tuple[float, float],
    polygon: np.ndarray,
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
            point,
            tuple(vertices[index]),
            tuple(vertices[(index + 1) % len(vertices)]),
        )
        for index in range(len(vertices))
    )


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
    return min(
        point_to_polygon_distance(corner, polygon) for corner in corners
    )


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
def bbox_containment_ratio(inner, outer) -> float:
    """计算 inner 框被 outer 框覆盖的比例（交集面积 / inner 面积）.

    与 bbox_iou 不同: 只关心 inner 是否被 outer 包住(如手框落在文件框内),
    不因 outer 框远超 inner 而衰减, 故指向文件的包含判定用本函数.

    Args:
        inner: 待判定包含关系的检测框 [x1, y1, x2, y2].
        outer: 可能包含 inner 的检测框 [x1, y1, x2, y2].

    Returns:
        inner 被 outer 覆盖的面积比例, 取值 0~1; inner 面积退化时返回 0.
    """
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    inner_area = max(0.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return intersection / inner_area if inner_area > 0 else 0.0


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
            point_a: 端点 A 坐标及置信度 [x, y, confidence]，或 [x, y].
            point_b: 顶点 B 坐标及置信度 [x, y, confidence]，或 [x, y].
            point_c: 端点 C 坐标及置信度 [x, y, confidence]，或 [x, y].

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
        cos_angle = float(
            np.clip(
                np.dot(vector_ba, vector_bc) / (norm_ba * norm_bc), -1.0, 1.0
            )
        )
        return float(np.degrees(np.arccos(cos_angle)))


class PoseEMAFilter:
    """关键点 EMA 平滑滤波器（对齐 behavior-v1）.

    对每个跟踪目标的关键点做指数移动平均，抑制单帧抖动；
    低置信度关键点（confidence < conf_threshold）沿用历史值，避免噪声误判。
    """

    def __init__(
        self, alpha: float = 0.5, conf_threshold: float = 0.25
    ) -> None:
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
            previous_x, previous_y, previous_confidence = previous_keypoints[
                keypoint_index
            ]
            if confidence < self.conf_threshold:
                smoothed[keypoint_index] = [
                    previous_x,
                    previous_y,
                    previous_confidence,
                ]
            else:
                smoothed[keypoint_index] = [
                    self.alpha * x + (1.0 - self.alpha) * previous_x,
                    self.alpha * y + (1.0 - self.alpha) * previous_y,
                    confidence,
                ]
        self.history[track_id] = smoothed.copy()
        return smoothed

    def remove(self, track_id: object) -> None:
        """移除指定跟踪目标的关键点历史缓存.

        Args:
            track_id: 外部跟踪标识；不存在时静默忽略.
        """
        self.history.pop(track_id, None)

    def clear(self) -> None:
        """清空全部跟踪目标的关键点历史缓存."""
        self.history.clear()


# 指向手势（MediaPipe 21 关键点，0 手腕; 1-4 拇指; 5-8 食指; 9-12 中指;
# 13-16 无名指; 17-20 小指）
FINGER_KEYPOINTS: Dict[str, Tuple[int, int, int]] = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}

# 镜像指序：MediaPipe 偶尔把指序标反，两种顺序都试一遍
FINGER_KEYPOINTS_MIRRORED: Dict[str, Tuple[int, int, int]] = {
    "index": (17, 18, 20),
    "middle": (13, 14, 16),
    "ring": (9, 10, 12),
    "pinky": (5, 6, 8),
}

KEYPOINT_INDEX_THUMB_TIP: int = 4
KEYPOINT_INDEX_MIDDLE_MCP: int = 9

ROLE_LEADER: str = "机组长"
ROLE_OPERATOR: str = "操作员"
ROLE_LEADER_AND_OPERATOR: str = "机组长和操作员"


def angle_at(
    endpoint_a: Sequence[float],
    vertex: Sequence[float],
    endpoint_c: Sequence[float],
) -> float:
    """计算 ∠(endpoint_a, vertex, endpoint_c) 的角度，单位度.

    与坐标尺度、旋转无关，适用于归一化或像素坐标。

    Args:
        endpoint_a: 端点 A 坐标 [x, y].
        vertex: 顶点坐标 [x, y].
        endpoint_c: 端点 C 坐标 [x, y].

    Returns:
        角度（度）; 任一向量退化时返回 0.0.
    """
    vector_ba = (endpoint_a[0] - vertex[0], endpoint_a[1] - vertex[1])
    vector_bc = (endpoint_c[0] - vertex[0], endpoint_c[1] - vertex[1])
    norm_ba = math.hypot(*vector_ba)
    norm_bc = math.hypot(*vector_bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosine = max(
        -1.0,
        min(
            1.0,
            (vector_ba[0] * vector_bc[0] + vector_ba[1] * vector_bc[1])
            / (norm_ba * norm_bc),
        ),
    )
    return math.degrees(math.acos(cosine))


def finger_angles(landmarks: np.ndarray) -> Dict[str, float]:
    """返回四根手指在 PIP 指节处的夹角.

    Args:
        landmarks: (21, 2) 关键点坐标.

    Returns:
        {index / middle / ring / pinky: PIP 夹角（度）}.
    """
    return {
        name: angle_at(landmarks[mcp], landmarks[pip], landmarks[tip])
        for name, (mcp, pip, tip) in FINGER_KEYPOINTS.items()
    }


def judge_pointing(
    landmarks: Optional[np.ndarray],
    straight_angle: float = 165.0,
    flex_angle: float = 160.0,
    straight_angle_relaxed: float = 130.0,
    point_gap: float = 35.0,
    try_mirrored_order: bool = True,
    point_thumb_gap: float = 0.60,
) -> Tuple[bool, Dict[str, Any]]:
    """判定 21 关键点是否构成「指向手势」.

    规则：无名指与小指均弯曲（PIP 夹角 < flex_angle），且食指或中指至少一根
    伸直。伸直满足任一即可：
      - 绝对条件：PIP 夹角 >= straight_angle；
      - 相对条件：PIP 夹角 >= straight_angle_relaxed，且比弯曲手指高
        point_gap 度（手指朝向镜头时被透视压缩，角度到不了 180 度）。
    另要求指向指尖离拇指尖足够远（point_thumb_gap），排除握笔等捏握动作。

    Args:
        landmarks: (21, 2) 关键点坐标，归一化或像素坐标均可（只用角度）.
        straight_angle: 判定伸直的绝对最小 PIP 夹角.
        flex_angle: 判定弯曲的最大 PIP 夹角.
        straight_angle_relaxed: 相对条件下允许的最小 PIP 夹角.
        point_gap: 相对条件下指向手指需比弯曲手指高出的角度.
        try_mirrored_order: 失败时是否按镜像指序再判一次.
        point_thumb_gap: 指向指尖到拇指尖距离与掌宽之比的下限.

    Returns:
        (是否指向, 判定信息字典); 信息字典含 reason 与四指角度，便于调参.
    """
    if landmarks is None or len(landmarks) < 21:
        return False, {"reason": "no_landmarks"}

    angles = finger_angles(landmarks)

    def thumb_gaps(
        finger_map: Dict[str, Tuple[int, int, int]],
    ) -> Dict[str, float]:
        """计算各指尖到拇指尖的距离与掌宽之比.

        Args:
            finger_map: 手指名到关键点三元组(mcp, pip, tip)的映射.

        Returns:
            手指名到(指尖-拇指尖距离)/掌宽比的映射.
        """
        palm_width = math.hypot(
            landmarks[KEYPOINT_INDEX_MIDDLE_MCP][0] - landmarks[0][0],
            landmarks[KEYPOINT_INDEX_MIDDLE_MCP][1] - landmarks[0][1],
        )
        return {
            name: math.hypot(
                landmarks[tip][0] - landmarks[KEYPOINT_INDEX_THUMB_TIP][0],
                landmarks[tip][1] - landmarks[KEYPOINT_INDEX_THUMB_TIP][1],
            )
            / max(palm_width, 1e-6)
            for name, (mcp, pip, tip) in finger_map.items()
        }

    def evaluate(
        angle_map: Dict[str, float], gap_map: Dict[str, float]
    ) -> Tuple[bool, str, bool, bool, bool, bool, float]:
        """按伸直 / 弯曲条件求值，返回是否指向与原因码.

        Args:
            angle_map: 各手指的关节角度.
            gap_map: 各指尖到拇指尖的距离/掌宽比.

        Returns:
            (是否指向, 原因码, 食指伸直, 中指伸直, 无名指弯曲, 小指弯曲, 参考角度).
        """
        reference_angle = max(angle_map["ring"], angle_map["pinky"])
        index_extended = angle_map["index"] >= straight_angle or (
            angle_map["index"] >= straight_angle_relaxed
            and angle_map["index"] - reference_angle >= point_gap
        )
        middle_extended = angle_map["middle"] >= straight_angle or (
            angle_map["middle"] >= straight_angle_relaxed
            and angle_map["middle"] - reference_angle >= point_gap
        )
        index_extended = index_extended and gap_map["index"] >= point_thumb_gap
        middle_extended = (
            middle_extended and gap_map["middle"] >= point_thumb_gap
        )
        ring_flexed = angle_map["ring"] < flex_angle
        pinky_flexed = angle_map["pinky"] < flex_angle

        if not (index_extended or middle_extended):
            reason = "pointing_finger_not_straight"
        elif not ring_flexed:
            reason = "ring_not_flexed"
        elif not pinky_flexed:
            reason = "pinky_not_flexed"
        else:
            reason = "ok"
        return (
            reason == "ok",
            reason,
            index_extended,
            middle_extended,
            ring_flexed,
            pinky_flexed,
            reference_angle,
        )

    gap_map = thumb_gaps(FINGER_KEYPOINTS)
    (
        is_pointing,
        reason,
        index_extended,
        middle_extended,
        ring_flexed,
        pinky_flexed,
        reference_angle,
    ) = evaluate(angles, gap_map)
    keypoint_order = "normal"
    if not is_pointing and try_mirrored_order:
        mirrored_angles = {
            name: angle_at(landmarks[mcp], landmarks[pip], landmarks[tip])
            for name, (mcp, pip, tip) in FINGER_KEYPOINTS_MIRRORED.items()
        }
        mirrored_gap_map = thumb_gaps(FINGER_KEYPOINTS_MIRRORED)
        (
            mirrored_pointing,
            _,
            mirrored_index,
            mirrored_middle,
            mirrored_ring,
            mirrored_pinky,
            mirrored_reference,
        ) = evaluate(mirrored_angles, mirrored_gap_map)
        if mirrored_pointing:
            is_pointing = True
            reason = "ok"
            keypoint_order = "mirrored"
            index_extended = mirrored_index
            middle_extended = mirrored_middle
            ring_flexed = mirrored_ring
            pinky_flexed = mirrored_pinky
            reference_angle = mirrored_reference
            gap_map = mirrored_gap_map

    info = {
        "reason": reason,
        "order": keypoint_order,
        "angles": {name: round(value, 1) for name, value in angles.items()},
        "thumb_gap": {
            name: round(value, 2) for name, value in gap_map.items()
        },
        "reference_angle": round(reference_angle, 1),
        "index_straight": index_extended,
        "middle_straight": middle_extended,
        "ring_flexed": ring_flexed,
        "pinky_flexed": pinky_flexed,
    }
    return is_pointing, info


class HandLandmarkExtractor:
    """对单个手框做「外扩 + 放大」后跑 MediaPipe Hands，返回 21 关键点."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """初始化 MediaPipe Hands 提取器（延迟导入，避免拖慢非行为进程）.

        Args:
            max_num_hands: 单次裁剪图内最多识别的手数.
            min_detection_confidence: 手部检测置信度阈值.
            min_tracking_confidence: 手部跟踪置信度阈值.
        """
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        """释放 MediaPipe 资源."""
        self._hands.close()

    def extract(
        self,
        frame: np.ndarray,
        bbox: Sequence[float],
        pad_ratio: float = 0.8,
        target_size: int = 480,
        other_bboxes: Optional[List[Sequence[float]]] = None,
    ) -> Optional[np.ndarray]:
        """提取手框内的 21 关键点（原图像素坐标）.

        Args:
            frame: 原图（BGR）.
            bbox: YOLO 手框 (x1, y1, x2, y2).
            pad_ratio: 外扩比例（相对手框宽 / 高，0.8 表示向外扩 0.8 倍）.
            target_size: 裁剪后长边目标像素，不足则放大.
            other_bboxes: 同帧其他手框，裁剪内对应区域涂黑避免认错手.

        Returns:
            (21, 2) float32 关键点（原图坐标）; 未检出返回 None.
        """
        import cv2

        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in bbox)
        bbox_width, bbox_height = x2 - x1, y2 - y1
        if bbox_width < 2 or bbox_height < 2:
            return None

        # 外扩：补回手腕 / 前臂上下文，MediaPipe 才能稳定检出
        offset_x = int(max(0, math.floor(x1 - bbox_width * pad_ratio)))
        offset_y = int(max(0, math.floor(y1 - bbox_height * pad_ratio)))
        crop_x_end = int(
            min(frame_width, math.ceil(x2 + bbox_width * pad_ratio))
        )
        crop_y_end = int(
            min(frame_height, math.ceil(y2 + bbox_height * pad_ratio))
        )
        if crop_x_end - offset_x < 16 or crop_y_end - offset_y < 16:
            return None

        # 放大：小手的识别率显著提升
        crop = frame[offset_y:crop_y_end, offset_x:crop_x_end]
        crop_height, crop_width = crop.shape[:2]
        scale = 1.0
        if target_size and max(crop_height, crop_width) < target_size:
            scale = min(4.0, target_size / max(crop_height, crop_width))
            crop = cv2.resize(
                crop,
                (
                    max(1, int(round(crop_width * scale))),
                    max(1, int(round(crop_height * scale))),
                ),
                interpolation=cv2.INTER_LINEAR,
            )
        input_height, input_width = crop.shape[:2]

        # 把同帧其他手框区域涂黑，避免 MediaPipe 认错手
        for other_bbox in other_bboxes or []:
            other_x1 = int(max(offset_x, other_bbox[0])) - offset_x
            other_y1 = int(max(offset_y, other_bbox[1])) - offset_y
            other_x2 = int(min(crop_x_end, other_bbox[2])) - offset_x
            other_y2 = int(min(crop_y_end, other_bbox[3])) - offset_y
            if other_x2 > other_x1 and other_y2 > other_y1:
                crop[
                    int(other_y1 * scale) : int(other_y2 * scale),
                    int(other_x1 * scale) : int(other_x2 * scale),
                ] = 0

        rgb_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb_image)
        if not result.multi_hand_landmarks:
            return None

        # 裁剪图内若有多只手，只取中心离 YOLO 框中心最近的那只
        reference_center_x = ((x1 + x2) / 2.0 - offset_x) * scale
        reference_center_y = ((y1 + y2) / 2.0 - offset_y) * scale
        best_points: Optional[np.ndarray] = None
        best_distance: Optional[float] = None
        for hand_landmarks in result.multi_hand_landmarks:
            points = np.array(
                [
                    [
                        point.x * input_width / scale + offset_x,
                        point.y * input_height / scale + offset_y,
                    ]
                    for point in hand_landmarks.landmark
                ],
                dtype=np.float32,
            )
            hand_center_x = (
                (points[:, 0].max() + points[:, 0].min()) / 2.0 - offset_x
            ) * scale
            hand_center_y = (
                (points[:, 1].max() + points[:, 1].min()) / 2.0 - offset_y
            ) * scale
            distance = math.hypot(
                hand_center_x - reference_center_x,
                hand_center_y - reference_center_y,
            )
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_points = points
        return best_points


class PoseRoleClassifier:
    """事件触发时刻的姿态角色判别器（手腕匹配 + 鼻尖高度比较）."""

    # COCO 关键点索引与置信度阈值
    NOSE_INDEX: int = 0
    LEFT_WRIST_INDEX: int = 9
    RIGHT_WRIST_INDEX: int = 10
    WRIST_CONFIDENCE_THRESHOLD: float = 0.3
    NOSE_CONFIDENCE_THRESHOLD: float = 0.18

    def __init__(self, pose_model_path: str) -> None:
        """初始化角色判别器并加载姿态模型.

        Args:
            pose_model_path: 姿态估计模型权重路径（YOLO-Pose）.
        """
        from ultralytics import YOLO

        logger.info(f"角色判别器加载姿态模型: {pose_model_path}")
        self._model = YOLO(pose_model_path)

    def classify(
        self,
        frame: np.ndarray,
        hand_bboxes: Sequence[Sequence[float]],
    ) -> Optional[str]:
        """对事件帧跑一次姿态推理，判别多个手框所属角色.

        Args:
            frame: 当前视频帧（BGR 格式）.
            hand_bboxes: 触发事件的手框列表，每项为 [x1, y1, x2, y2].

        Returns:
            '机组长' / '操作员' / '机组长和操作员'；无法判定时返回 None.
        """
        results = self._model(frame, verbose=False)
        if not results or results[0].keypoints is None:
            return None
        keypoints_data = results[0].keypoints.data
        if keypoints_data is None or len(keypoints_data) == 0:
            return None

        keypoints = keypoints_data.cpu().numpy()
        person_count = len(keypoints)

        # 收集可见手腕（记录其所属人体索引）
        wrists: List[Tuple[int, float, float]] = []
        for person_index in range(person_count):
            left_wrist = keypoints[person_index][self.LEFT_WRIST_INDEX]
            right_wrist = keypoints[person_index][self.RIGHT_WRIST_INDEX]
            if left_wrist[2] >= self.WRIST_CONFIDENCE_THRESHOLD:
                wrists.append(
                    (person_index, float(left_wrist[0]), float(left_wrist[1]))
                )
            if right_wrist[2] >= self.WRIST_CONFIDENCE_THRESHOLD:
                wrists.append(
                    (
                        person_index,
                        float(right_wrist[0]),
                        float(right_wrist[1]),
                    )
                )
        if not wrists:
            return None

        # 鼻尖 y 升序：y 最小（头最高）= 机组长
        noses: List[Tuple[int, float]] = []
        for person_index in range(person_count):
            nose = keypoints[person_index][self.NOSE_INDEX]
            if nose[2] >= self.NOSE_CONFIDENCE_THRESHOLD:
                noses.append((person_index, float(nose[1])))
        noses.sort(key=lambda item: item[1])

        # 画面内仅一人时默认机组长（机组长站后排，会遮挡操作员）
        if person_count <= 1:
            return ROLE_LEADER

        leader_index = noses[0][0] if noses else -1
        role_set = set()
        for hand_bbox in hand_bboxes:
            hand_center_x = (float(hand_bbox[0]) + float(hand_bbox[2])) / 2.0
            hand_center_y = (float(hand_bbox[1]) + float(hand_bbox[3])) / 2.0

            best_distance = float("inf")
            best_person_index = -1
            for person_index, wrist_x, wrist_y in wrists:
                distance = math.hypot(
                    hand_center_x - wrist_x, hand_center_y - wrist_y
                )
                if distance < best_distance:
                    best_distance = distance
                    best_person_index = person_index

            if best_person_index >= 0:
                if best_person_index == leader_index:
                    role_set.add(ROLE_LEADER)
                else:
                    role_set.add(ROLE_OPERATOR)

        if not role_set:
            return None
        if len(role_set) == 1:
            return role_set.pop()
        return ROLE_LEADER_AND_OPERATOR
