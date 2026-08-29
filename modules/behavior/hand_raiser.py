"""举手动作检测器.

基于 modules.behavior.behavior_utils 提供的姿态平滑、几何计算与空间匹配工具，
结合连续帧积分累加器与冷却去重机制，实现高精度、无抖动的举手行为识别。
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.behavior.behavior_utils import PoseEMAFilter, PoseTools, bbox_iou
from modules.tracker.multi_object_tracker import STrack

logger = logging.getLogger("module.behavior.hand_raiser")


def is_hand_up(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    bbox_height: float,
    conf_threshold: float = 0.25,
) -> bool:
    """判断单臂是否处于举手状态.

    几何规则（对齐 behavior-v1）:
    1. 肩膀与手腕关键点置信度需达到阈值；
    2. 手腕垂直位置高于肩膀基准线（结合人体高度自适应比例阈值）；
    3. 若肘部可见，手臂夹角 (shoulder-elbow-wrist) 需 >= 50 度，避免自然下垂或搭手误判。

    Args:
        shoulder: 肩膀关键点坐标及置信度 [x, y, conf].
        elbow: 肘部关键点坐标及置信度 [x, y, conf].
        wrist: 手腕关键点坐标及置信度 [x, y, conf].
        bbox_height: 姿态检测边界框高度（像素）.
        conf_threshold: 关键点有效置信度阈值，默认 0.25.

    Returns:
        满足举手几何特征返回 True，否则返回 False.
    """
    if shoulder[2] < conf_threshold or wrist[2] < conf_threshold:
        return False
    elbow_visible = elbow[2] >= conf_threshold
    ratio_thresh = bbox_height * (0.06 if elbow_visible else 0.12)
    if wrist[1] > shoulder[1] - max(ratio_thresh, 15.0):
        return False
    if elbow_visible:
        angle = PoseTools.calc_angle(shoulder, elbow, wrist)
        if angle is not None and angle < 50.0:
            return False
    return True


class HandRaiser:
    """举手动作检测器（实现归属 behavior 模块）.

    检测流程：
    1. 接收跟踪模块传入的前置视角视频帧与多目标跟踪轨迹（List[STrack]）；
    2. 使用 YOLO-Pose 模型检测画面中所有人体骨架关键点；
    3. 采用水平中心对齐与 bbox_iou 综合评分匹配 MOT 全身框与 Pose 姿态框；
    4. 使用 PoseEMAFilter 对匹配的人体关键点执行指数移动平均平滑滤波；
    5. 对左右双臂分别执行 is_hand_up 几何特征判定；
    6. 以 track_id 为键维护连续帧积分累加器（consec_raise 与 consec_idle）及冷却期；
    7. 返回当前帧触发举手的目标元组列表 [(track_id, identity), ...]供驱动方调用。
    """

    def __init__(
        self,
        detector: Any,
        consec_raise: int = 2,
        consec_idle: int = 3,
        cooldown_frames: int = 45,
        frame_step: int = 3,
    ) -> None:
        """初始化举手检测器.

        Args:
            detector: 目标检测器，包含 detect_pose 方法.
            consec_raise: 判定举手所需的连续累加得分阈值，默认 2.
            consec_idle: 退出举手所需的连续静息得分阈值，默认 3.
            cooldown_frames: 触发举手后的冷却帧数（如 1.5s * 30fps = 45帧）.
            frame_step: 姿态推理跳帧步长，每 frame_step 帧推理一次，其余帧复用
                上次结果（对齐 behavior-v1）；姿态变化慢 + EMA 平滑 + 连续累计，稀疏不影响判定.
        """
        self._detector: Any = detector
        self._consec_raise: int = consec_raise
        self._consec_idle: int = consec_idle
        self._cooldown_frames: int = cooldown_frames
        self._frame_step: int = max(1, int(frame_step))
        self._last_poses: Optional[List[Dict[str, Any]]] = None

        self._pose_filter: PoseEMAFilter = PoseEMAFilter(
            alpha=0.5, conf_threshold=0.25
        )
        self._raise_state: Dict[int, Dict[str, Any]] = {}
        self._event_cooldown: Dict[int, int] = {}
        self._last_seen_frame: Dict[int, int] = {}

    def check(
        self,
        frame: np.ndarray,
        tracks: List[STrack],
        frame_count: int,
    ) -> List[Tuple[int, Optional[str]]]:
        """对当前帧所有跟踪目标执行举手动作检测.

        Args:
            frame: 原始视频帧 (BGR 格式).
            tracks: 当前帧的多目标跟踪轨迹列表.
            frame_count: 当前视频帧号.

        Returns:
            触发举手的目标列表 [(track_id, identity), ...]。
            若已分配工位身份，identity 为 'LEADER'|'ROAD1'|'ROAD2'；若未分配则为 None。
        """
        if not tracks:
            return []

        # 1. 姿态检测: 每 frame_step 帧推理一次, 其余帧复用上次结果(对齐 behavior-v1)
        if frame_count % self._frame_step == 0 or self._last_poses is None:
            poses = self._detector.detect_pose(frame)
            self._last_poses = poses
        else:
            poses = self._last_poses
        if not poses:
            return []

        raised_targets: List[Tuple[int, Optional[str]]] = []

        # 2. 遍历跟踪目标
        for track in tracks:
            track_id = track.track_id
            self._last_seen_frame[track_id] = frame_count
            track_bbox = track.bbox

            # 3. 空间匹配：使用水平中心对齐与 bbox_iou 综合评分匹配 MOT 全身框与 Pose 姿态框
            best_pose: Optional[Dict[str, Any]] = None
            best_score = 0.0
            track_bbox_width = max(1.0, float(track_bbox[2] - track_bbox[0]))
            track_center_x = (float(track_bbox[0]) + float(track_bbox[2])) / 2.0

            for pose in poses:
                pose_bbox = pose["box"]
                iou = bbox_iou(track_bbox, pose_bbox)
                pose_center_x = (float(pose_bbox[0]) + float(pose_bbox[2])) / 2.0
                center_distance = abs(track_center_x - pose_center_x)

                if center_distance < track_bbox_width * 0.8:
                    score = iou * 2.0 + (1.0 - center_distance / track_bbox_width)
                    if score > best_score:
                        best_score = score
                        best_pose = pose

            if best_pose is None:
                continue

            # 4. 关键点 EMA 平滑滤波
            raw_keypoints = best_pose["keypoints"]
            keypoints = self._pose_filter.update(track_id, raw_keypoints)
            best_pose_bbox = best_pose["box"]
            pose_bbox_height = float(best_pose_bbox[3] - best_pose_bbox[1])

            # 5. 左右臂举手几何判定（COCO 关键点：左 5/7/9、右 6/8/10 为肩/肘/腕）
            left_arm_raised = is_hand_up(
                keypoints[5], keypoints[7], keypoints[9], pose_bbox_height
            )
            right_arm_raised = is_hand_up(
                keypoints[6], keypoints[8], keypoints[10], pose_bbox_height
            )
            is_hand_raised = left_arm_raised or right_arm_raised

            # 6. 时序连续得分累加器
            if track_id not in self._raise_state:
                self._raise_state[track_id] = {"score": 0, "raised": False}
            raise_state = self._raise_state[track_id]

            if is_hand_raised:
                raise_state["score"] += 1
            else:
                raise_state["score"] = max(raise_state["score"] - 1, -self._consec_idle)

            if raise_state["score"] >= self._consec_raise and not raise_state["raised"]:
                raise_state["raised"] = True
            elif raise_state["score"] <= -self._consec_idle and raise_state["raised"]:
                raise_state["raised"] = False

            # 7. 冷却期判定与事件触发
            frames_since_last_event = (
                frame_count - self._event_cooldown.get(track_id, -99999)
            )
            if raise_state["raised"] and frames_since_last_event > self._cooldown_frames:
                self._event_cooldown[track_id] = frame_count
                raised_targets.append((track_id, track.identity))

        # 8. 定期清理长期未见轨迹的内部状态缓存
        if frame_count % 600 == 0:
            self._cleanup_expired_tracks(frame_count)

        return raised_targets

    def _cleanup_expired_tracks(
        self, current_frame: int, expire_frames: int = 600
    ) -> None:
        """清理已消失轨迹的内部缓存，防止内存泄漏.

        Args:
            current_frame: 当前帧号.
            expire_frames: 判定过期的未见帧数阈值，默认 600 帧.
        """
        expired_track_ids = [
            track_id for track_id, last_frame in self._last_seen_frame.items()
            if current_frame - last_frame > expire_frames
        ]
        for track_id in expired_track_ids:
            self._raise_state.pop(track_id, None)
            self._event_cooldown.pop(track_id, None)
            self._last_seen_frame.pop(track_id, None)
            self._pose_filter.remove(track_id)

    def reset(self) -> None:
        """重置内部所有状态与缓存."""
        self._raise_state.clear()
        self._event_cooldown.clear()
        self._last_seen_frame.clear()
        self._pose_filter.clear()
        self._last_poses = None
