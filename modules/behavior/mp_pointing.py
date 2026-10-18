"""YOLO 四分类 + MediaPipe Hand 的指向手势检测器（移植自 behavior-v2/mp_pointing.py）.

检测流程:
  1. YOLO 四分类(hand / file / person / pointing_hand) 检出候选手框与文件框,
     可选并集模型（三分类）补手框召回;
  2. 对每个候选手框: 外扩补回手腕/前臂 → 同帧其他手框区域涂黑 → 放大到目标长边,
     裁剪后送 MediaPipe Hands 取 21 关键点并映射回原图坐标;
  3. 指向判定: 无名指与小指弯曲, 且食指或中指伸直（绝对角或相对角）,
     并要求指向指尖离拇指尖足够远, 排除握笔等捏握; 失败时按镜像指序再判一次;
  4. 与四分类结果分级融合: pointing_hand 高置信直接采信, 中间置信需规则复核,
     hand 类由 MediaPipe 规则决定;
  5. 目标归属: 手框被 file 框包含 → 指向文件; 与机箱多边形重叠比/距离达标 → 手指屏幕;
  6. 事件去重: 指向文件按 role 合并(merge_seconds) + 遮挡宽限(grace_frames),
     手指屏幕上升沿触发 + debounce_frames 防抖 + cooldown_sec 冷却.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from modules.behavior.behavior_utils import (
    bbox_containment_ratio,
    bbox_iou,
    bbox_polygon_overlap_ratio,
    bbox_to_polygon_distance,
)
from modules.behavior.behavior_vis import (
    SCREEN_POLYGONS,
    draw_class_boxes,
    draw_hand_landmarks,
    draw_trigger,
)
from modules.behavior.pose_role import PoseRoleClassifier

logger = logging.getLogger("module.behavior.mp_pointing")

# 事件类型标识（与事件总线 topic 映射一致）
EVENT_FINGER_FILE: str = "FINGER_FILE"
EVENT_FINGER_SCREEN: str = "FINGER_SCREEN"
STATE_FINGER_FILE: str = "手指指向文件"
STATE_FINGER_SCREEN: str = "手指指向屏幕"

# MediaPipe 21 关键点索引（标准约定）
#   0 手腕; 1-4 拇指; 5-8 食指; 9-12 中指; 13-16 无名指; 17-20 小指
KP_INDEX_MCP, KP_INDEX_PIP, KP_INDEX_TIP = 5, 6, 8
KP_MIDDLE_MCP, KP_MIDDLE_PIP, KP_MIDDLE_TIP = 9, 10, 12
KP_RING_MCP, KP_RING_PIP, KP_RING_TIP = 13, 14, 16
KP_PINKY_MCP, KP_PINKY_PIP, KP_PINKY_TIP = 17, 18, 20

# 每根手指: (根部 MCP, 中间指节 PIP, 指尖 TIP)
FINGERS: Dict[str, Tuple[int, int, int]] = {
    "index": (KP_INDEX_MCP, KP_INDEX_PIP, KP_INDEX_TIP),
    "middle": (KP_MIDDLE_MCP, KP_MIDDLE_PIP, KP_MIDDLE_TIP),
    "ring": (KP_RING_MCP, KP_RING_PIP, KP_RING_TIP),
    "pinky": (KP_PINKY_MCP, KP_PINKY_PIP, KP_PINKY_TIP),
}

# 镜像指序: MediaPipe 偶尔把指序标反, 两种顺序都试一遍
FINGERS_MIRROR: Dict[str, Tuple[int, int, int]] = {
    "index": (KP_PINKY_MCP, KP_PINKY_PIP, KP_PINKY_TIP),
    "middle": (KP_RING_MCP, KP_RING_PIP, KP_RING_TIP),
    "ring": (KP_MIDDLE_MCP, KP_MIDDLE_PIP, KP_MIDDLE_TIP),
    "pinky": (KP_INDEX_MCP, KP_INDEX_PIP, KP_INDEX_TIP),
}

# 机箱多边形基准分辨率（SCREEN_POLYGONS 按 1920x1080 标注）
BASE_WIDTH, BASE_HEIGHT = 1920, 1080


def angle_at(
    point_a: Sequence[float],
    point_b: Sequence[float],
    point_c: Sequence[float],
) -> float:
    """计算 ∠ABC（顶点为 point_b）的角度, 单位度, 与坐标尺度/旋转无关.

    Args:
        point_a: 端点 A 坐标 [x, y].
        point_b: 顶点 B 坐标 [x, y].
        point_c: 端点 C 坐标 [x, y].

    Returns:
        角度（度）; 任一向量退化时返回 0.0.
    """
    vector_ba = (point_a[0] - point_b[0], point_a[1] - point_b[1])
    vector_bc = (point_c[0] - point_b[0], point_c[1] - point_b[1])
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
        {index/middle/ring/pinky: PIP 夹角（度）}.
    """
    return {
        name: angle_at(landmarks[mcp], landmarks[pip], landmarks[tip])
        for name, (mcp, pip, tip) in FINGERS.items()
    }


def scale_polygons(
    polygons: Sequence[Sequence[Sequence[float]]],
    width: int,
    height: int,
    base: Tuple[int, int] = (BASE_WIDTH, BASE_HEIGHT),
) -> List[List[Tuple[int, int]]]:
    """把基准分辨率下标注的机箱多边形按实际视频分辨率等比缩放.

    Args:
        polygons: 机箱多边形列表, 每项为顶点序列.
        width: 实际帧宽.
        height: 实际帧高.
        base: 标注基准分辨率.

    Returns:
        缩放后的多边形列表（顶点取整）.
    """
    scale_x, scale_y = width / base[0], height / base[1]
    if abs(scale_x - 1.0) < 1e-3 and abs(scale_y - 1.0) < 1e-3:
        return [list(polygon) for polygon in polygons]
    return [
        [
            (int(round(x * scale_x)), int(round(y * scale_y)))
            for x, y in polygon
        ]
        for polygon in polygons
    ]


def judge_pointing(
    landmarks: Optional[np.ndarray],
    straight_angle: float = 160.0,
    flex_angle: float = 160.0,
    straight_relaxed: float = 130.0,
    point_gap: float = 25.0,
    try_mirrored: bool = True,
    point_thumb_gap: float = 0.55,
) -> Tuple[bool, Dict[str, Any]]:
    """判定 21 关键点是否构成「指向手势」.

    规则: 无名指与小指均弯曲(PIP 夹角 < flex_angle), 且食指或中指至少一根伸直。
    伸直满足任一即可:
      - 绝对条件: PIP 夹角 >= straight_angle;
      - 相对条件: PIP 夹角 >= straight_relaxed 且比弯曲手指(ref)高出 point_gap 度
        （手指朝向镜头/桌面时被透视压缩, 角度到不了 180°, 用相对条件兜住）.
    另要求指向指尖离拇指尖足够远（point_thumb_gap）, 排除握笔等捏握。

    Args:
        landmarks: (21, 2) 关键点坐标, 归一化或像素坐标均可（只用角度, 尺度无关）.
        straight_angle: 判定「伸直」的绝对最小 PIP 夹角.
        flex_angle: 判定「弯曲」的最大 PIP 夹角.
        straight_relaxed: 相对条件下允许的最小 PIP 夹角.
        point_gap: 相对条件下指向手指需比弯曲手指高出的角度.
        try_mirrored: 失败时是否按镜像指序再判一次.
        point_thumb_gap: 指向指尖到拇指尖距离 / 掌宽 的下限.

    Returns:
        (是否指向, 判定信息字典); 信息字典含 reason 与四指角度, 便于调参与可视化.
    """
    if landmarks is None or len(landmarks) < 21:
        return False, {"reason": "no_landmarks"}

    angles = finger_angles(landmarks)

    def thumb_gaps(order: Dict[str, Tuple[int, int, int]]) -> Dict[str, float]:
        """计算各指尖到拇指尖的距离与掌宽的比值."""
        palm_width = math.hypot(
            landmarks[9][0] - landmarks[0][0],
            landmarks[9][1] - landmarks[0][1],
        )
        return {
            name: math.hypot(
                landmarks[tip][0] - landmarks[4][0],
                landmarks[tip][1] - landmarks[4][1],
            )
            / max(palm_width, 1e-6)
            for name, (mcp, pip, tip) in order.items()
        }

    def evaluate(angle_map: Dict[str, float], gap_map: Dict[str, float]):
        """按伸直/弯曲条件求值, 返回是否指向与原因码."""
        reference = max(angle_map["ring"], angle_map["pinky"])
        index_extended = angle_map["index"] >= straight_angle or (
            angle_map["index"] >= straight_relaxed
            and angle_map["index"] - reference >= point_gap
        )
        middle_extended = angle_map["middle"] >= straight_angle or (
            angle_map["middle"] >= straight_relaxed
            and angle_map["middle"] - reference >= point_gap
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
            (reason == "ok"),
            reason,
            index_extended,
            middle_extended,
            ring_flexed,
            pinky_flexed,
            reference,
        )

    gap = thumb_gaps(FINGERS)
    (
        ok,
        reason,
        index_extended,
        middle_extended,
        ring_flexed,
        pinky_flexed,
        reference,
    ) = evaluate(angles, gap)
    order = "normal"
    if not ok and try_mirrored:
        mirror_angles = {
            name: angle_at(landmarks[a], landmarks[b], landmarks[c])
            for name, (a, b, c) in FINGERS_MIRROR.items()
        }
        mirror_gap = thumb_gaps(FINGERS_MIRROR)
        (
            mirror_ok,
            _,
            mir_index,
            mir_middle,
            mir_ring,
            mir_pinky,
            mir_reference,
        ) = evaluate(mirror_angles, mirror_gap)
        if mirror_ok:
            ok, reason, order = True, "ok", "mirrored"
            index_extended, middle_extended = mir_index, mir_middle
            ring_flexed, pinky_flexed, reference = (
                mir_ring,
                mir_pinky,
                mir_reference,
            )
            gap = mirror_gap

    info = {
        "reason": reason,
        "order": order,
        "angles": {name: round(value, 1) for name, value in angles.items()},
        "thumb_gap": {name: round(value, 2) for name, value in gap.items()},
        "ref": round(reference, 1),
        "index_straight": index_extended,
        "middle_straight": middle_extended,
        "ring_flexed": ring_flexed,
        "pinky_flexed": pinky_flexed,
    }
    return ok, info


class HandLandmarkExtractor:
    """对单个手框做「外扩 + 放大」后跑 MediaPipe Hands, 返回 21 关键点."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """初始化 MediaPipe Hands 提取器.

        Args:
            max_num_hands: 单次裁剪图内最多识别的手数.
            min_detection_confidence: 手部检测置信度阈值.
            min_tracking_confidence: 手部跟踪置信度阈值.
        """
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
        pad_ratio: float = 1.0,
        target_size: int = 480,
        max_upscale: float = 4.0,
        others: Optional[List[Sequence[float]]] = None,
    ) -> Optional[np.ndarray]:
        """提取手框内的 21 关键点（原图像素坐标）.

        Args:
            frame: 原图 (BGR).
            bbox: YOLO 手框 (x1, y1, x2, y2).
            pad_ratio: 外扩比例（相对手框宽/高, 1.0 表示向外扩一倍）.
            target_size: 裁剪后长边目标像素, 不足则放大.
            max_upscale: 最大放大倍数, 防止小框放大到爆内存.
            others: 同帧其他手框, 裁剪内对应区域涂黑避免认错手.

        Returns:
            (21, 2) float32 关键点（原图坐标）; 未检出返回 None.
        """
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in bbox)
        box_width, box_height = x2 - x1, y2 - y1
        if box_width < 2 or box_height < 2:
            return None

        # 1) 外扩: 补回手腕/前臂上下文, MediaPipe 才能稳定检出
        offset_x = int(max(0, math.floor(x1 - box_width * pad_ratio)))
        offset_y = int(max(0, math.floor(y1 - box_height * pad_ratio)))
        crop_x2 = int(min(frame_width, math.ceil(x2 + box_width * pad_ratio)))
        crop_y2 = int(
            min(frame_height, math.ceil(y2 + box_height * pad_ratio))
        )
        if crop_x2 - offset_x < 16 or crop_y2 - offset_y < 16:
            return None

        # 2) 放大: 小手的识别率显著提升
        crop = frame[offset_y:crop_y2, offset_x:crop_x2]
        crop_height, crop_width = crop.shape[:2]
        scale = 1.0
        if target_size and max(crop_height, crop_width) < target_size:
            scale = min(
                max_upscale, target_size / max(crop_height, crop_width)
            )
            crop = cv2.resize(
                crop,
                (
                    max(1, int(round(crop_width * scale))),
                    max(1, int(round(crop_height * scale))),
                ),
                interpolation=cv2.INTER_LINEAR,
            )
        input_height, input_width = crop.shape[:2]

        # 2b) 把同帧其他手框区域涂黑, 避免 MediaPipe 认错手
        for other_box in others or []:
            other_x1 = int(max(offset_x, other_box[0])) - offset_x
            other_y1 = int(max(offset_y, other_box[1])) - offset_y
            other_x2 = int(min(crop_x2, other_box[2])) - offset_x
            other_y2 = int(min(crop_y2, other_box[3])) - offset_y
            if other_x2 > other_x1 and other_y2 > other_y1:
                crop[
                    int(other_y1 * scale) : int(other_y2 * scale),
                    int(other_x1 * scale) : int(other_x2 * scale),
                ] = 0

        # 3) 推理
        rgb_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb_image)
        if not result.multi_hand_landmarks:
            return None

        # 4) 裁剪图内若有多只手, 只取「中心离 YOLO 框中心最近」的那只
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
                best_distance, best_points = distance, points
        return best_points


class MPPointingDetector:
    """YOLO 四分类 + MediaPipe Hand 的「指向文件 / 手指屏幕」检测器."""

    def __init__(
        self,
        weights_path: str,
        pose_model_path: str,
        config: Optional[Dict[str, Any]] = None,
        union_weights_path: Optional[str] = None,
        device: str = "0",
    ) -> None:
        """初始化指向检测器（加载 YOLO 四分类与 MediaPipe）.

        Args:
            weights_path: YOLO 四分类权重路径（hand/file/person/pointing_hand）.
            pose_model_path: 角色判别用姿态模型路径.
            config: behavior.mp_pointing 子配置, 缺省时取内置默认值.
            union_weights_path: 可选并集模型（三分类）权重, 用于补手框召回.
            device: YOLO 推理设备标识, 取值 "0" / "cpu" 等.
        """
        mode_config = dict(config or {})
        self._weights_path = weights_path
        self._pose_model_path = pose_model_path
        self._device = str(mode_config.get("device", device))

        self.conf = float(mode_config.get("conf", 0.25))
        self.frame_step = max(1, int(mode_config.get("frame_step", 3)))
        self.pad_ratio = float(mode_config.get("pad_ratio", 0.8))
        self.target_size = int(mode_config.get("target_size", 480))
        self.max_num_hands = int(mode_config.get("max_num_hands", 2))
        self.straight_angle = float(mode_config.get("straight_angle", 165.0))
        self.flex_angle = float(mode_config.get("flex_angle", 160.0))
        self.straight_relaxed = float(
            mode_config.get("straight_angle_relaxed", 130.0)
        )
        self.point_gap = float(mode_config.get("point_gap", 35.0))
        self.try_mirrored = bool(mode_config.get("try_mirrored_order", True))
        self.point_thumb_gap = float(mode_config.get("point_thumb_gap", 0.60))

        # 四分类置信度分级融合
        self.p4_conf_high = float(mode_config.get("p4_conf_hi", 0.5))
        self.p4_conf_low = float(mode_config.get("p4_conf_lo", 0.25))
        self.other_hand_iou = float(mode_config.get("other_hand_iou", 0.2))
        self.union_iou = float(mode_config.get("union_iou", 0.5))

        # 目标归属
        self.file_contain_threshold = float(
            mode_config.get("file_contain_threshold", 0.7)
        )
        self.screen_overlap_threshold = float(
            mode_config.get("screen_overlap_threshold", 0.2)
        )
        self.max_distance = float(mode_config.get("max_dist", 10.0))

        # 事件去重
        self.merge_seconds = float(mode_config.get("merge_seconds", 5.0))
        self.role_cooldown_seconds = float(
            mode_config.get("role_cooldown_seconds", 1.5)
        )
        self.grace_frames = int(mode_config.get("grace_frames", 12))
        self.cooldown_seconds = float(mode_config.get("cooldown_sec", 1.5))
        self.debounce_frames = int(mode_config.get("debounce_frames", 3))
        self.draw_all_hands = bool(mode_config.get("draw_all_hands", True))

        logger.info(f"加载 YOLO 四分类模型: {self._weights_path}")
        self._model = YOLO(self._weights_path)
        self._class_hand = self._class_index_by_name("hand", 0)
        self._class_file = self._class_index_by_name("file", 1)
        self._class_person = self._class_index_by_name("person", 2)
        self._class_pointing_hand = self._class_index_by_name(
            "pointing_hand", -1
        )
        self._candidate_classes = {self._class_hand}
        if self._class_pointing_hand >= 0:
            self._candidate_classes.add(self._class_pointing_hand)
        logger.info(f"模型类别: {getattr(self._model, 'names', {})}")

        self._union_model: Optional[YOLO] = None
        if union_weights_path:
            logger.info(f"加载并集召回模型: {union_weights_path}")
            self._union_model = YOLO(union_weights_path)

        self._extractor = HandLandmarkExtractor(
            max_num_hands=self.max_num_hands,
            min_detection_confidence=float(
                mode_config.get("mp_detection_conf", 0.3)
            ),
        )
        # 角色判别器按需懒加载(事件触发时才需要姿态模型)
        self._pose_role: Optional[PoseRoleClassifier] = None

        self._polygons: Optional[List[List[Tuple[int, int]]]] = None
        self._last_detections: Optional[Dict[str, Any]] = None

        # 指向文件: 合并式去重状态
        self._last_file_frame: int = -99999
        self._last_file_role: Optional[str] = None
        self._last_file_event_frame: int = -99999
        self._last_file_event_role: Optional[str] = None
        # 手指屏幕: 上升沿 + 防抖状态
        self._screen_active: bool = False
        self._screen_idle_frames: int = 0
        self._last_screen_event_frame: int = -99999

    def _class_index_by_name(self, class_name: str, default_index: int) -> int:
        """按类名解析模型中的类别索引（类名缺失时回退默认索引）.

        Args:
            class_name: 目标类别名, 如 hand.
            default_index: 未匹配到时的回退索引.

        Returns:
            类别索引; 回退时返回 default_index.
        """
        names = getattr(self._model, "names", {}) or {}
        for index, name in names.items():
            if str(name).lower() == class_name:
                return int(index)
        return default_index

    def _ensure_polygons(
        self, frame: np.ndarray
    ) -> List[List[Tuple[int, int]]]:
        """按首帧尺寸缩放出机箱多边形并缓存.

        Args:
            frame: 当前视频帧.

        Returns:
            与当前帧分辨率匹配的机箱多边形.
        """
        if self._polygons is None:
            height, width = frame.shape[:2]
            self._polygons = scale_polygons(SCREEN_POLYGONS, width, height)
            logger.info(
                f"机箱多边形已按 {width}x{height} 缩放: {len(self._polygons)} 个"
            )
        return self._polygons

    def _role_of(
        self, frame: np.ndarray, hands: List[Dict[str, Any]]
    ) -> Optional[str]:
        """事件触发时判别人手角色（懒加载姿态模型）.

        Args:
            frame: 事件发生时的视频帧.
            hands: 触发事件的手部结果列表.

        Returns:
            '机组长' / '操作员' / '机组长和操作员'；不可判定时返回 None.
        """
        if not hands:
            return None
        if self._pose_role is None:
            self._pose_role = PoseRoleClassifier(self._pose_model_path)
        try:
            return self._pose_role.classify_batch(
                frame, [hand["bbox"] for hand in hands]
            )
        except Exception as role_error:  # 角色判别失败不应中断检测
            logger.warning(f"角色判别失败: {role_error}")
            return None

    def _fuse(
        self,
        is_pointing_class: bool,
        class_confidence: float,
        rule_pointing: bool,
    ) -> bool:
        """四分类结果与 MediaPipe 规则的分级融合.

        - 四分类判为 pointing_hand 且置信度 >= p4_conf_hi → 直接采信（MediaPipe 检不出也认）;
        - 置信度在 [p4_conf_lo, p4_conf_hi) → 需 MediaPipe 规则也判为指向;
        - 置信度 < p4_conf_lo → 丢弃;
        - 非 pointing_hand（hand 类）→ 由 MediaPipe 规则决定.

        Args:
            is_pointing_class: 该框是否来自 pointing_hand 类.
            class_confidence: 该框的检测置信度.
            rule_pointing: MediaPipe 规则判定结果.

        Returns:
            融合后是否判定为指向手.
        """
        if is_pointing_class:
            if class_confidence >= self.p4_conf_high:
                return True
            if class_confidence >= self.p4_conf_low:
                return rule_pointing
            return False
        return rule_pointing

    def _detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """跑一次 YOLO（可多模型并集） + MediaPipe, 返回本帧检测与融合判决.

        Args:
            frame: 当前视频帧.

        Returns:
            检测结果字典, 含 detections / files / hands_all / pointing_hands.
        """
        result = self._model.predict(
            frame, conf=self.conf, device=self._device, verbose=False
        )[0]
        detections: List[Dict[str, Any]] = []
        names = getattr(self._model, "names", {}) or {}
        for index in range(len(result.boxes)):
            class_index = int(result.boxes.cls[index])
            detections.append(
                {
                    "cls": class_index,
                    "name": names.get(class_index, str(class_index)),
                    "bbox": tuple(
                        int(value)
                        for value in result.boxes.xyxy[index].cpu().numpy()
                    ),
                    "conf": float(result.boxes.conf[index]),
                }
            )

        # 可选: 再跑一个手框召回更好的模型, 把它的 hand 框并进来（去重）
        if self._union_model is not None:
            union_result = self._union_model.predict(
                frame, conf=self.conf, device=self._device, verbose=False
            )[0]
            existing_boxes = [item["bbox"] for item in detections]
            for index in range(len(union_result.boxes)):
                if int(union_result.boxes.cls[index]) != self._class_hand:
                    continue
                union_box = tuple(
                    int(value)
                    for value in union_result.boxes.xyxy[index].cpu().numpy()
                )
                if any(
                    bbox_iou(union_box, box) >= self.union_iou
                    for box in existing_boxes
                ):
                    continue
                detections.append(
                    {
                        "cls": self._class_hand,
                        "name": "hand",
                        "bbox": union_box,
                        "conf": float(union_result.boxes.conf[index]),
                    }
                )
                existing_boxes.append(union_box)

        files = [
            item["bbox"]
            for item in detections
            if item["cls"] == self._class_file
        ]
        candidates = [
            item
            for item in detections
            if item["cls"] in self._candidate_classes
        ]
        candidate_boxes = [item["bbox"] for item in candidates]

        hands_all: List[Dict[str, Any]] = []
        pointing_hands: List[Dict[str, Any]] = []
        for candidate in candidates:
            others = [
                box
                for box in candidate_boxes
                if box != candidate["bbox"]
                and bbox_iou(box, candidate["bbox"]) < self.other_hand_iou
            ]
            landmarks = self._extractor.extract(
                frame,
                candidate["bbox"],
                pad_ratio=self.pad_ratio,
                target_size=self.target_size,
                others=others,
            )
            if landmarks is None:
                rule_pointing, info = False, {"reason": "mp_no_hand"}
            else:
                rule_pointing, info = judge_pointing(
                    landmarks,
                    self.straight_angle,
                    self.flex_angle,
                    self.straight_relaxed,
                    self.point_gap,
                    self.try_mirrored,
                    self.point_thumb_gap,
                )
            is_pointing_class = candidate["cls"] == self._class_pointing_hand
            is_pointing = self._fuse(
                is_pointing_class, candidate["conf"], rule_pointing
            )

            info = dict(info)
            info.update(
                {
                    "p4": is_pointing_class,
                    "conf4": round(candidate["conf"], 3),
                    "pm": rule_pointing,
                    "fused": is_pointing,
                }
            )
            hand = {
                "bbox": candidate["bbox"],
                "landmarks": landmarks,
                "info": info,
                "pointing": is_pointing,
                "target": None,
            }
            if landmarks is not None:
                hands_all.append(hand)
            if is_pointing:
                pointing_hands.append(hand)

        return {
            "detections": detections,
            "files": files,
            "hands_all": hands_all,
            "pointing_hands": pointing_hands,
        }

    def _target_of(
        self,
        hand_bbox: Sequence[float],
        files: List[Sequence[float]],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """判断指向目标: file / screen / None.

        优先级: 先判「指向文件」（手框被文件框包含）, 再判「手指屏幕」（机箱区域重叠或贴近）.

        Args:
            hand_bbox: 指向手框 [x1, y1, x2, y2].
            files: 本帧文件框列表.

        Returns:
            (目标类型, 判定信息); 目标类型取 "file" / "screen" / None.
        """
        best_contain_ratio = 0.0
        for file_box in files:
            best_contain_ratio = max(
                best_contain_ratio, bbox_containment_ratio(hand_bbox, file_box)
            )
        if best_contain_ratio >= self.file_contain_threshold:
            return "file", {"contain_ratio": round(best_contain_ratio, 3)}

        for polygon in self._polygon_contours():
            overlap = bbox_polygon_overlap_ratio(hand_bbox, polygon)
            distance = bbox_to_polygon_distance(hand_bbox, polygon)
            if (
                overlap > self.screen_overlap_threshold
                or distance < self.max_distance
            ):
                return "screen", {
                    "overlap": round(overlap, 3),
                    "dist": round(distance, 1),
                }

        return None, {"contain_ratio": round(best_contain_ratio, 3)}

    def _polygon_contours(self) -> List[np.ndarray]:
        """把当前机箱多边形转为 OpenCV 轮廓数组.

        Returns:
            int32 轮廓数组列表.
        """
        polygons = (
            self._polygons if self._polygons is not None else SCREEN_POLYGONS
        )
        return [np.array(polygon, dtype=np.int32) for polygon in polygons]

    def _should_log_file_event(
        self, role: Optional[str], frame_count: int, fps: float
    ) -> bool:
        """指向文件事件的合并式去重判定.

        规则: 与最近一次已记录事件比较——
          同 role 且间隔 <= merge_seconds  → 合并（不记录）;
          role 不同且间隔 <= role_cooldown_seconds → 视为 role 抖动, 合并（不记录）;
          否则记录新事件.

        Args:
            role: 当前判定的行为人角色.
            frame_count: 当前帧号.
            fps: 视频帧率.

        Returns:
            需要记录新事件时返回 True.
        """
        if self._last_file_event_frame >= 0:
            frame_gap = frame_count - self._last_file_event_frame
            same_role = role == self._last_file_event_role
            merge_frames = max(1, int(self.merge_seconds * fps))
            cooldown_frames = max(1, int(self.role_cooldown_seconds * fps))
            if same_role and frame_gap <= merge_frames:
                return False
            if (not same_role) and frame_gap <= cooldown_frames:
                return False
        self._last_file_event_frame = frame_count
        self._last_file_event_role = role
        return True

    def _should_log_screen_event(self, frame_count: int, fps: float) -> bool:
        """手指屏幕事件的冷却去重判定.

        Args:
            frame_count: 当前帧号.
            fps: 视频帧率.

        Returns:
            脱离冷却期时返回 True.
        """
        cooldown_frames = max(1, int(self.cooldown_seconds * fps))
        if frame_count - self._last_screen_event_frame <= cooldown_frames:
            return False
        self._last_screen_event_frame = frame_count
        return True

    @staticmethod
    def _make_event(
        event_type: str,
        state: str,
        frame_count: int,
        fps: float,
        role: Optional[str],
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构造标准化的行为事件字典.

        Args:
            event_type: 事件标识（FINGER_FILE / FINGER_SCREEN）.
            state: 行为状态文案.
            frame_count: 事件发生帧号.
            fps: 视频帧率, 用于换算 localSec.
            role: 行为人角色, 不可判定时为 None.
            extra: 额外判定信息.

        Returns:
            标准化事件字典.
        """
        event = {
            "event": event_type,
            "state": state,
            "frame_id": frame_count,
            "localSec": round(frame_count / fps, 2),
            "role": role,
            "operator": role,
        }
        event.update(extra)
        return event

    def process_frame(
        self,
        frame: np.ndarray,
        frame_count: int,
        fps: float,
    ) -> List[Dict[str, Any]]:
        """处理单帧: 跳帧复用 → 目标归属 → 事件去重 → 标注绘制.

        Args:
            frame: 当前视频帧（原地绘制检测框与关键点）.
            frame_count: 当前帧号（从 1 开始计数）.
            fps: 视频帧率.

        Returns:
            本帧新触发的事件列表, 每项含 event / state / localSec / role 等字段.
        """
        self._ensure_polygons(frame)

        # 跳帧: 非关键帧复用上一次检测结果, 保持事件连续性
        if frame_count % self.frame_step != 1:
            if self._last_detections is None:
                return []
            detections = self._last_detections
        else:
            detections = self._detect(frame)
            self._last_detections = detections

        events: List[Dict[str, Any]] = []
        # 目标归属: 每个指向手都打上 target 标记, 供可视化与调试
        file_hands: List[Dict[str, Any]] = []
        screen_hands: List[Dict[str, Any]] = []
        for pointing_hand in detections["pointing_hands"]:
            target, target_info = self._target_of(
                pointing_hand["bbox"], detections["files"]
            )
            pointing_hand["target"] = target
            pointing_hand["target_info"] = target_info
            if target == "file":
                file_hands.append(pointing_hand)
            elif target == "screen":
                screen_hands.append(pointing_hand)

        # ---- 指向文件: 合并式记录（同 role 且间隔 <= merge_seconds 视为同一次）----
        if file_hands:
            role = self._role_of(frame, file_hands)
            self._last_file_frame = frame_count
            self._last_file_role = role
            if self._should_log_file_event(role, frame_count, fps):
                best_info = file_hands[0].get("target_info", {})
                events.append(
                    self._make_event(
                        EVENT_FINGER_FILE,
                        STATE_FINGER_FILE,
                        frame_count,
                        fps,
                        role,
                        {"contain_ratio": best_info.get("contain_ratio", 0.0)},
                    )
                )
        elif frame_count - self._last_file_frame <= self.grace_frames:
            # 遮挡宽限: 短暂漏检仍视为持续指向, 顺延合并窗口避免同一动作被切成多段
            if self._last_file_role is not None:
                self._should_log_file_event(
                    self._last_file_role, frame_count, fps
                )

        # ---- 手指屏幕: 上升沿触发（持续指向只记一次）----
        if screen_hands:
            self._screen_idle_frames = 0
            if not self._screen_active:
                self._screen_active = True
                role = self._role_of(frame, screen_hands)
                if self._should_log_screen_event(frame_count, fps):
                    best_info = screen_hands[0].get("target_info", {})
                    events.append(
                        self._make_event(
                            EVENT_FINGER_SCREEN,
                            STATE_FINGER_SCREEN,
                            frame_count,
                            fps,
                            role,
                            {
                                "overlap_ratio": best_info.get("overlap", 0.0),
                                "min_dist": best_info.get("dist", 0.0),
                            },
                        )
                    )
        else:
            self._screen_idle_frames += 1
            if self._screen_idle_frames >= self.debounce_frames:
                self._screen_active = False

        self._draw(frame, detections, file_hands, screen_hands)
        return events

    def _draw(
        self,
        frame: np.ndarray,
        detections: Dict[str, Any],
        file_hands: List[Dict[str, Any]],
        screen_hands: List[Dict[str, Any]],
    ) -> None:
        """把检测框、手部关键点与事件触发标记画进帧（原地）.

        Args:
            frame: 当前视频帧.
            detections: 本帧检测结果.
            file_hands: 触发指向文件的手.
            screen_hands: 触发手指屏幕的手.
        """
        draw_class_boxes(frame, detections.get("detections"))

        marked_hands = list(detections.get("hands_all") or [])
        if not self.draw_all_hands:
            marked_hands = [
                hand for hand in marked_hands if hand.get("pointing")
            ]
        draw_hand_landmarks(frame, marked_hands, draw_all_hands=True)

        if file_hands:
            draw_trigger(
                frame, [hand["bbox"] for hand in file_hands], "FINGER_FILE!"
            )
        if screen_hands:
            draw_trigger(
                frame,
                [hand["bbox"] for hand in screen_hands],
                "FINGER_SCREEN!",
            )

    def close(self) -> None:
        """释放 MediaPipe 与模型资源."""
        try:
            self._extractor.close()
        except Exception as close_error:
            logger.warning(f"释放 MediaPipe 失败: {close_error}")
