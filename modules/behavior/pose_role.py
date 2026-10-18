"""事件触发时刻的姿态角色判别（移植自 behavior-v2/pose_role.py）.

仅在行为事件（指向屏幕 / 指向文件）触发时对当前帧跑一次姿态模型:
  1. 取所有人体骨架的手腕关键点(conf 达标), 与触发事件的手框中心做距离匹配, 确定手的主人;
  2. 取所有人体骨架的鼻尖 y 坐标排序, y 最小(头最高)=机组长, 其余=操作员;
  3. 只有一个人时默认机组长(机组长站后排, 会遮挡操作员);
  4. 多人同时指向时合并为 "机组长和操作员".
"""

import logging
import math
from typing import Any, List, Optional, Sequence, Tuple

from ultralytics import YOLO

logger = logging.getLogger("module.behavior.pose_role")

# COCO 关键点索引
KP_NOSE: int = 0
KP_LEFT_WRIST: int = 9
KP_RIGHT_WRIST: int = 10

# 关键点有效置信度阈值
WRIST_CONF_THRESHOLD: float = 0.3
NOSE_CONF_THRESHOLD: float = 0.18

# 角色文案（与 behavior-v2 一致）
ROLE_LEADER: str = "机组长"
ROLE_OPERATOR: str = "操作员"
ROLE_BOTH: str = "机组长和操作员"


class PoseRoleClassifier:
    """事件触发时的角色判别器（姿态模型 + 手腕匹配 + 鼻尖高度比较）."""

    def __init__(self, pose_model_path: str) -> None:
        """初始化角色判别器并加载姿态模型.

        Args:
            pose_model_path: 姿态估计模型权重路径（YOLO-Pose）.
        """
        logger.info(f"角色判别器加载姿态模型: {pose_model_path}")
        self._model = YOLO(pose_model_path)

    def classify_batch(
        self,
        frame: Any,
        hand_bboxes: Sequence[Sequence[float]],
    ) -> Optional[str]:
        """对事件帧跑一次姿态推理, 判别多个手框所属角色.

        Args:
            frame: 当前视频帧 (BGR 格式).
            hand_bboxes: 触发事件的手框列表, 每项为 [x1, y1, x2, y2].

        Returns:
            角色字符串 '机组长' / '操作员' / '机组长和操作员'；无法判定时返回 None.
        """
        results = self._model(frame, verbose=False)
        if not results or results[0].keypoints is None:
            return None

        keypoints_data = results[0].keypoints.data
        if keypoints_data is None or len(keypoints_data) == 0:
            return None

        keypoints = keypoints_data.cpu().numpy()
        person_count = len(keypoints)

        # 1. 收集所有可见手腕(记录其所属人体索引)
        wrists: List[Tuple[int, float, float]] = []
        for person_index in range(person_count):
            left_wrist = keypoints[person_index][KP_LEFT_WRIST]
            right_wrist = keypoints[person_index][KP_RIGHT_WRIST]
            if left_wrist[2] >= WRIST_CONF_THRESHOLD:
                wrists.append(
                    (person_index, float(left_wrist[0]), float(left_wrist[1]))
                )
            if right_wrist[2] >= WRIST_CONF_THRESHOLD:
                wrists.append(
                    (
                        person_index,
                        float(right_wrist[0]),
                        float(right_wrist[1]),
                    )
                )

        if not wrists:
            return None

        # 2. 收集可见鼻尖 y 坐标并升序排序(y 小 = 头高 = 机组长)
        noses: List[Tuple[int, float]] = []
        for person_index in range(person_count):
            nose = keypoints[person_index][KP_NOSE]
            if nose[2] >= NOSE_CONF_THRESHOLD:
                noses.append((person_index, float(nose[1])))
        noses.sort(key=lambda item: item[1])

        # 3. 画面内仅一人 → 默认机组长
        if person_count <= 1:
            return ROLE_LEADER

        top_person_index = noses[0][0] if noses else -1

        # 4. 每个手框匹配最近手腕, 收集角色集合
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
                if best_person_index == top_person_index:
                    role_set.add(ROLE_LEADER)
                else:
                    role_set.add(ROLE_OPERATOR)

        # 5. 合并角色
        if not role_set:
            return None
        if len(role_set) == 1:
            return role_set.pop()
        return ROLE_BOTH
