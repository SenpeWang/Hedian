"""行为检测模块入口.

在单一 pop 视角（camPOP.mpg）上运行一次 YOLO 四分类（hand / file / person /
pointing_hand）检测，对候选手框提取 MediaPipe 21 关键点并判定指向手势，
再分别交给手指指向屏幕（FingerScreenDetector）与手指指向文件
（FingerFileDetector）两个判定器归属与去重；
举手检测由 tracker 模块在前置视角调用 HandRaiser，本模块通过 BehaviorStorage
共享事件落盘。
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from core.base_module import BaseModule
from core.event_bus import EventBus, EventTopic
from core.inference_stream import InferenceStream
from core.path_manager import PathConfig
from core.vis_encoder import VisEncoder

from modules.behavior.behavior_utils import (
    HandLandmarkExtractor,
    PoseRoleClassifier,
    bbox_iou,
    judge_pointing,
)
from modules.behavior.behavior_vis import (
    draw_class_boxes,
    draw_hand_landmarks,
    draw_roi_overlay,
    scale_screen_polygons,
)
from modules.behavior.file_detector import FingerFileDetector
from modules.behavior.screen_detect import FingerScreenDetector
from modules.behavior.storage_behavior import BehaviorStorage

logger = logging.getLogger("module.behavior")

_EVENT_TOPIC_MAP: Dict[str, EventTopic] = {
    "FINGER_SCREEN": EventTopic.BEHAVIOR_FINGER_SCREEN,
    "FINGER_FILE": EventTopic.BEHAVIOR_FINGER_FILE,
}

# 默认模型路径（相对项目根, 可被 config.behavior.detection 覆盖）
_DEFAULT_WEIGHTS = "models/behavior/detect_4class.pt"
_DEFAULT_POSE_MODEL = "models/behavior/pose.pt"
_DEFAULT_UNION_WEIGHTS = "models/behavior/detect_3class_union.pt"


class BehaviorModule(BaseModule):
    """行为检测模块.

    管理行为事件（手指屏幕、手指文件、举手）的推理、推送与结果落盘。
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: Dict[str, Any],
        paths: PathConfig,
        inference_stream: InferenceStream,
    ) -> None:
        """初始化行为检测模块.

        Args:
            event_bus: 全局事件总线.
            config: 全局配置字典.
            paths: 路径管理器.
            inference_stream: 前端推理流推送通道.
        """
        super().__init__(event_bus, config, paths, inference_stream)
        self._model: Optional[YOLO] = None
        self._union_model: Optional[YOLO] = None
        self._extractor: Optional[HandLandmarkExtractor] = None
        self._role_classifier: Optional[PoseRoleClassifier] = None
        self._screen_detector: Optional[FingerScreenDetector] = None
        self._file_detector: Optional[FingerFileDetector] = None
        self._result_storage: Optional[BehaviorStorage] = None
        self._vis_encoder: Optional[VisEncoder] = None
        self._events: List[Dict[str, Any]] = []
        self._events_lock: threading.Lock = threading.Lock()
        self._last_progress_push: float = 0.0
        self._screen_polygons: List[Tuple[str, np.ndarray]] = []

    @property
    def module_name(self) -> str:
        """获取模块名称.

        Returns:
            模块标识字符串 'behavior'.
        """
        return "behavior"

    def _resolve_path(self, configured_path: str, default_path: str) -> str:
        """把配置中的相对模型路径解析为绝对路径（相对项目根）.

        Args:
            configured_path: 配置里的路径, 空串时回退默认值.
            default_path: 默认相对路径.

        Returns:
            绝对路径字符串.
        """
        path = configured_path or default_path
        if path and not path.startswith("/"):
            path = str(self.paths.base_dir / path)
        return path

    def initialize(self) -> bool:
        """初始化检测模型、指向判定器、角色判别器与举手事件订阅.

        Returns:
            初始化成功返回 True，失败返回 False.
        """
        try:
            behavior_config = self.config.get("behavior", {})
            detection_config = behavior_config.get("detection", {})
            screen_config = behavior_config.get("finger_screen", {})
            file_config = behavior_config.get("finger_file", {})

            fps = self._read_video_fps("pop")

            weights_path = self._resolve_path(
                str(detection_config.get("weights", "")), _DEFAULT_WEIGHTS
            )
            pose_model_path = self._resolve_path(
                str(detection_config.get("pose_model", "")),
                _DEFAULT_POSE_MODEL,
            )
            union_configured = detection_config.get(
                "union_weights", _DEFAULT_UNION_WEIGHTS
            )
            union_weights_path = (
                self._resolve_path(
                    str(union_configured), _DEFAULT_UNION_WEIGHTS
                )
                if union_configured
                else ""
            )

            self._device = str(detection_config.get("device", "0"))
            self._confidence = float(detection_config.get("confidence", 0.25))
            self._frame_step = max(
                1, int(detection_config.get("frame_step", 8))
            )
            self._pad_ratio = float(detection_config.get("pad_ratio", 0.8))
            self._target_size = int(detection_config.get("target_size", 480))
            self._other_hand_iou = float(
                detection_config.get("other_hand_iou", 0.2)
            )
            self._union_iou = float(detection_config.get("union_iou", 0.5))
            self._draw_all_hands = bool(
                detection_config.get("draw_all_hands", True)
            )
            self._four_class_confidence_hi = float(
                detection_config.get("four_class_confidence_hi", 0.5)
            )
            self._four_class_confidence_lo = float(
                detection_config.get("four_class_confidence_lo", 0.25)
            )
            self._judge_kwargs = {
                "straight_angle": float(
                    detection_config.get("straight_angle", 165.0)
                ),
                "flex_angle": float(detection_config.get("flex_angle", 160.0)),
                "straight_angle_relaxed": float(
                    detection_config.get("straight_angle_relaxed", 130.0)
                ),
                "point_gap": float(detection_config.get("point_gap", 35.0)),
                "try_mirrored_order": bool(
                    detection_config.get("try_mirrored_order", True)
                ),
                "point_thumb_gap": float(
                    detection_config.get("point_thumb_gap", 0.60)
                ),
            }

            logger.info(f"加载行为检测模型: {weights_path}")
            self._model = YOLO(weights_path)
            self._class_hand = self._class_index_by_name("hand", 0)
            self._class_file = self._class_index_by_name("file", 1)
            self._class_pointing_hand = self._class_index_by_name(
                "pointing_hand", -1
            )
            self._candidate_classes = {self._class_hand}
            if self._class_pointing_hand >= 0:
                self._candidate_classes.add(self._class_pointing_hand)

            if union_weights_path:
                logger.info(f"加载手框并集召回模型: {union_weights_path}")
                self._union_model = YOLO(union_weights_path)

            self._extractor = HandLandmarkExtractor(
                max_num_hands=int(detection_config.get("max_num_hands", 2)),
                min_detection_confidence=float(
                    detection_config.get("mp_detection_confidence", 0.3)
                ),
            )
            self._role_classifier = PoseRoleClassifier(pose_model_path)

            self._screen_detector = FingerScreenDetector(
                screen_overlap_threshold=float(
                    screen_config.get("screen_overlap_threshold", 0.2)
                ),
                max_distance=int(screen_config.get("max_dist", 10)),
                cooldown_sec=float(screen_config.get("cooldown_sec", 1.5)),
                fps=fps,
                debounce_frames=int(screen_config.get("debounce_frames", 3)),
            )
            self._file_detector = FingerFileDetector(
                file_contain_threshold=float(
                    file_config.get("file_contain_threshold", 0.7)
                ),
                merge_sec=float(file_config.get("merge_sec", 5.0)),
                role_cooldown_sec=float(
                    file_config.get("role_cooldown_sec", 1.5)
                ),
                grace_frames=int(file_config.get("grace_frames", 12)),
                fps=fps,
            )

            self._result_storage = BehaviorStorage(self.paths)
            self.event_bus.subscribe(
                EventTopic.BEHAVIOR_HAND_RAISED, self._on_hand_raised
            )

            logger.info("行为检测模块初始化完成")
            return True

        except Exception as init_error:
            logger.error(
                f"行为检测模块初始化失败: {init_error}", exc_info=True
            )
            return False

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

    def _read_video_fps(self, video_key: str) -> float:
        """读取配置中指定视角视频的真实帧率, 读不到直接报错(不兜底).

        Args:
            video_key: 视频键名（如 'pop'）.

        Returns:
            视频真实帧率.

        Raises:
            RuntimeError: 视频无法打开或帧率读取失败.
        """
        videos_config = self.config.get("videos", {})
        rel_path = videos_config.get(
            video_key, f"data/videos/cam{video_key.upper()}.mpg"
        )
        abs_path = str(self.paths.base_dir / rel_path)
        cap = cv2.VideoCapture(abs_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"无法打开视频读取帧率 ({video_key}): {abs_path}"
            )
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not fps or fps <= 0:
            raise RuntimeError(
                f"读取视频帧率失败 ({video_key}, CAP_PROP_FPS={fps}): "
                f"{abs_path}"
            )
        return float(fps)

    def _append_event(
        self,
        local_sec: float,
        key_moment: str,
        role: Optional[str] = None,
    ) -> None:
        """线程安全地收集关键事件 + 实时落盘.

        Args:
            local_sec: 事件发生的时间戳（秒）.
            key_moment: 关键时刻描述文本.
            role: 行为人角色，不可判定时为 None.
        """
        event: Dict[str, Any] = {
            "localSec": round(local_sec, 2),
            "key_moment": key_moment,
            "source": "behavior",
        }
        if role:
            event["role"] = role
        with self._events_lock:
            self._events.append(event)
        # 实时落盘(不等到 save_results), 保证评估提取时文件有全部事件
        if self._result_storage and self._run_id:
            try:
                self._result_storage.add_event(self._run_id, event)
            except Exception as disk_error:
                logger.warning(f"实时落盘行为事件失败: {disk_error}")

    def _on_hand_raised(self, event: Dict[str, Any]) -> None:
        """订阅 BEHAVIOR_HAND_RAISED 事件的回调函数.

        Args:
            event: 事件总线消息字典，业务字段位于 "data" 子字典.
        """
        payload = event.get("data", {})
        operator_name = payload.get("operator") or event.get("operator")
        timestamp = payload.get("localSec", event.get("ts", 0))
        logger.info(f"行为模块收到举手事件: {operator_name} @{timestamp:.1f}s")

    def _fuse_pointing(
        self,
        is_pointing_class: bool,
        confidence: float,
        rule_pointing: bool,
    ) -> bool:
        """四分类结果与 MediaPipe 手势规则的分级融合.

        - 四分类判为 pointing_hand 且置信度达标 → 直接采信;
        - 置信度处于过渡区间 → 需 MediaPipe 规则也判为指向;
        - 非 pointing_hand（hand 类）→ 由 MediaPipe 规则决定.

        Args:
            is_pointing_class: 该框是否来自 pointing_hand 类.
            confidence: 该框的检测置信度.
            rule_pointing: MediaPipe 手势规则判定结果.

        Returns:
            融合后是否判定为指向手.
        """
        if not is_pointing_class:
            return rule_pointing
        if confidence >= self._four_class_confidence_hi:
            return True
        if confidence >= self._four_class_confidence_lo:
            return rule_pointing
        return False

    def _detect_hands(self, frame: Any) -> Dict[str, Any]:
        """跑一次 YOLO 四分类（可并集召回） + MediaPipe，返回手框与指向判定.

        Args:
            frame: 当前视频帧.

        Returns:
            检测结果字典，含 detections / file_bboxes / hand_marks /
            pointing_bboxes.
        """
        result = self._model.predict(
            frame,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )[0]
        names = getattr(self._model, "names", {}) or {}
        detections: List[Dict[str, Any]] = []
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
                    "confidence": float(result.boxes.conf[index]),
                }
            )

        if self._union_model is not None:
            union_result = self._union_model.predict(
                frame,
                conf=self._confidence,
                device=self._device,
                verbose=False,
            )[0]
            existing_bboxes = [item["bbox"] for item in detections]
            for index in range(len(union_result.boxes)):
                if int(union_result.boxes.cls[index]) != self._class_hand:
                    continue
                union_bbox = tuple(
                    int(value)
                    for value in union_result.boxes.xyxy[index].cpu().numpy()
                )
                if any(
                    bbox_iou(union_bbox, bbox) >= self._union_iou
                    for bbox in existing_bboxes
                ):
                    continue
                detections.append(
                    {
                        "cls": self._class_hand,
                        "name": "hand",
                        "bbox": union_bbox,
                        "confidence": float(union_result.boxes.conf[index]),
                    }
                )
                existing_bboxes.append(union_bbox)

        file_bboxes = [
            item["bbox"]
            for item in detections
            if item["cls"] == self._class_file
        ]
        candidates = [
            item
            for item in detections
            if item["cls"] in self._candidate_classes
        ]
        candidate_bboxes = [item["bbox"] for item in candidates]

        hand_marks: List[Dict[str, Any]] = []
        pointing_bboxes: List[Sequence[float]] = []
        for candidate in candidates:
            other_bboxes = [
                bbox
                for bbox in candidate_bboxes
                if bbox != candidate["bbox"]
                and bbox_iou(bbox, candidate["bbox"]) < self._other_hand_iou
            ]
            landmarks = self._extractor.extract(
                frame,
                candidate["bbox"],
                pad_ratio=self._pad_ratio,
                target_size=self._target_size,
                other_bboxes=other_bboxes,
            )
            if landmarks is None:
                rule_pointing = False
            else:
                rule_pointing, _ = judge_pointing(
                    landmarks, **self._judge_kwargs
                )
            is_pointing = self._fuse_pointing(
                candidate["cls"] == self._class_pointing_hand,
                candidate["confidence"],
                rule_pointing,
            )
            hand_marks.append(
                {
                    "bbox": candidate["bbox"],
                    "landmarks": landmarks,
                    "pointing": is_pointing,
                }
            )
            if is_pointing:
                pointing_bboxes.append(candidate["bbox"])

        return {
            "detections": detections,
            "file_bboxes": file_bboxes,
            "hand_marks": hand_marks,
            "pointing_bboxes": pointing_bboxes,
        }

    def _process_pop_view(self, video_path: str, video_source: str) -> None:
        """在 pop 视角上运行指向检测（YOLO 四分类 + MediaPipe）.

        Args:
            video_path: pop 视频文件路径.
            video_source: 推理流进度 source 名称.
        """
        try:
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                logger.error(f"[pop_view] 无法打开视频: {video_path}")
                return

            raw_fps = video_capture.get(cv2.CAP_PROP_FPS)
            if not raw_fps or raw_fps <= 0:
                raise RuntimeError(
                    f"读取 pop 视频帧率失败(CAP_PROP_FPS={raw_fps}), 不再兜底"
                )
            fps = float(raw_fps)
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            total_sec = (total_frames / fps) if total_frames > 0 else 0.0
            frame_count = 0
            cached_hands: Optional[Dict[str, Any]] = None

            logger.info(
                f"[pop_view] 开始处理: {video_path} "
                f"({total_frames}帧, {fps:.1f}fps, 跳帧 {self._frame_step})"
            )

            # 视觉流编码器:pop 视角(无音频) -> fMP4 推 Redis Stream
            self._vis_encoder = VisEncoder(
                view="pop",
                video_path=video_path,
                fps=fps,
                with_audio=False,
                redis_host=self.config.get("redis_host", "localhost"),
                redis_port=self.config.get("redis_port", 6379),
                redis_db=self.config.get("redis_db", 0),
            )

            while True:
                frame_read_success, frame = video_capture.read()
                if not frame_read_success:
                    break

                frame_count += 1
                timestamp = frame_count / fps

                self.inference_stream.update_module_time(
                    video_source, timestamp
                )
                if total_frames > 0:
                    now = time.time()
                    if now - self._last_progress_push >= 0.3:
                        self._last_progress_push = now
                        progress_percentage = min(
                            100.0, timestamp / total_sec * 100
                        )
                        self.push_display(
                            "progress",
                            {
                                "localSec": round(timestamp, 2),
                                "tag": "progress",
                                "data": {
                                    "label": "pop",
                                    "pct": round(progress_percentage, 1),
                                },
                            },
                        )

                # 机箱多边形按首帧尺寸缩放一次，推理与绘制共用
                if not self._screen_polygons:
                    frame_height, frame_width = frame.shape[:2]
                    self._screen_polygons = scale_screen_polygons(
                        frame_width, frame_height
                    )

                # 跳帧：非关键帧复用上次检测结果，保持事件连续性
                if frame_count % self._frame_step == 1 or cached_hands is None:
                    cached_hands = self._detect_hands(frame)

                # 角色判别需在干净帧上跑：标注会遮挡人体关键点，降低姿态检出率
                role_frame = frame.copy()

                # ROI 底层 -> 检测框 -> 手部骨架，均为原地绘制
                draw_roi_overlay(frame, self._screen_polygons)
                draw_class_boxes(frame, cached_hands["detections"])
                draw_hand_landmarks(
                    frame,
                    cached_hands["hand_marks"],
                    draw_all_hands=self._draw_all_hands,
                )

                events: List[Dict[str, Any]] = []
                events += self._screen_detector.detect(
                    frame,
                    cached_hands["pointing_bboxes"],
                    self._screen_polygons,
                    frame_count,
                    self._role_classifier,
                    role_frame,
                )
                events += self._file_detector.detect(
                    frame,
                    cached_hands["pointing_bboxes"],
                    cached_hands["file_bboxes"],
                    frame_count,
                    self._role_classifier,
                    role_frame,
                )

                # 带标注帧喂入视觉流编码器 -> fMP4 -> 前端 MSE
                if self._vis_encoder is not None:
                    try:
                        self._vis_encoder.feed_frame(frame)
                    except Exception as vis_error:
                        logger.warning(f"pop 喂帧失败: {vis_error}")

                for event in events:
                    event_sec = round(event.get("localSec", timestamp), 2)
                    role = event.get("role")
                    self._append_event(
                        event_sec, event.get("state", "behavior"), role
                    )

                    payload = {
                        "localSec": event_sec,
                        "tag": event.get("event", "behavior"),
                        "data": {
                            key: value
                            for key, value in event.items()
                            if key != "localSec"
                        },
                    }
                    self.push_display("behavior", payload)
                    topic = _EVENT_TOPIC_MAP.get(event.get("event", ""))
                    if topic is None:
                        logger.warning(
                            f"未知行为事件类型: {event.get('event')}，"
                            "跳过事件流推送"
                        )
                        continue
                    self.push_event(topic, payload, timestamp=event_sec)

                if frame_count % 300 == 0:
                    progress_percentage = (
                        frame_count * 100 // total_frames
                        if total_frames
                        else 0
                    )
                    logger.info(
                        f"[pop_view] {frame_count}/{total_frames}帧 "
                        f"{progress_percentage}%"
                    )

            video_capture.release()
            if self._vis_encoder is not None:
                self._vis_encoder.finalize()
            logger.info(f"[pop_view] 完成，共 {frame_count} 帧")

        except Exception as pop_error:
            logger.error(
                f"[pop_view] 视频处理失败: {pop_error}", exc_info=True
            )

    def process_video(self, video_path: str) -> None:
        """在单一 pop 视角上处理指向检测.

        Args:
            video_path: 视频文件路径（可被配置项覆盖）.
        """
        videos_config = self.config.get("videos", {})
        base_dir = self.paths.base_dir
        pop_video = str(
            base_dir / videos_config.get("pop", "data/videos/camPOP.mpg")
        )

        logger.info(
            f"pop 视角指向检测（YOLO 四分类 + MediaPipe）: {pop_video}"
        )
        self._process_pop_view(pop_video, "video_pop")
        logger.info(f"pop 视角处理完成，共 {len(self._events)} 条事件")

    def save_results(self, run_id: str) -> None:
        """收尾兜底保存: 实时落盘已逐条写入, 此处仅合并补写缺失事件(去重).

        Args:
            run_id: 本次运行标识.
        """
        with self._events_lock:
            events_list = list(self._events)
        self._result_storage.merge_events(run_id, events_list)
        if self._extractor is not None:
            self._extractor.close()
