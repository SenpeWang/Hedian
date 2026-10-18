"""行为检测模块入口（指向检测采用 behavior-v2 的 MediaPipe 方案）.

在单一 pop 视角（camPOP.mpg）上运行「YOLO 四分类 + MediaPipe 21 关键点」指向检测器,
统一判别手指指向屏幕（FingerScreen）与手指指向文件（FingerFile）并附带行为人角色;
举手检测由 tracker 模块在前置视角调用 HandRaiser, 本模块通过 BehaviorStorage 共享事件落盘。
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import cv2

from core.base_module import BaseModule
from core.event_bus import EventBus, EventTopic
from core.inference_stream import InferenceStream
from core.path_manager import PathConfig
from core.vis_encoder import VisEncoder

from modules.behavior.behavior_vis import (
    draw_roi_overlay,
    screen_polygons_as_tuples,
)
from modules.behavior.mp_pointing import MPPointingDetector
from modules.behavior.storage_behavior import BehaviorStorage

logger = logging.getLogger("module.behavior")

_EVENT_TOPIC_MAP: Dict[str, EventTopic] = {
    "FINGER_SCREEN": EventTopic.BEHAVIOR_FINGER_SCREEN,
    "FINGER_FILE": EventTopic.BEHAVIOR_FINGER_FILE,
}

# 默认模型路径（相对项目根, 可被 config.behavior.mp_pointing 覆盖）
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
        self._pointing_detector: Optional[MPPointingDetector] = None
        self._result_storage: Optional[BehaviorStorage] = None
        self._vis_encoder: Optional[VisEncoder] = None
        self._events: List[Dict[str, Any]] = []
        self._events_lock: threading.Lock = threading.Lock()
        self._last_progress_push: float = 0.0

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
        """初始化指向检测器与举手事件订阅.

        Returns:
            初始化成功返回 True，失败返回 False.
        """
        try:
            pointing_config = self.config.get("behavior", {}).get(
                "mp_pointing", {}
            )
            weights_path = self._resolve_path(
                str(pointing_config.get("weights", "")), _DEFAULT_WEIGHTS
            )
            pose_model_path = self._resolve_path(
                str(pointing_config.get("pose_model", "")), _DEFAULT_POSE_MODEL
            )
            union_config = pointing_config.get(
                "union_weights", _DEFAULT_UNION_WEIGHTS
            )
            union_weights_path = (
                self._resolve_path(str(union_config), _DEFAULT_UNION_WEIGHTS)
                if union_config
                else None
            )

            self._pointing_detector = MPPointingDetector(
                weights_path=weights_path,
                pose_model_path=pose_model_path,
                config=pointing_config,
                union_weights_path=union_weights_path,
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

    def _append_event(
        self,
        local_sec: float,
        key_moment: str,
        role: Optional[str] = None,
    ) -> None:
        """线程安全地收集关键事件 + 实时落盘(保证评估提取时文件有全部事件).

        Args:
            local_sec: 事件发生的时间戳（秒）.
            key_moment: 关键时刻描述文本.
            role: 行为人角色（机组长/操作员/机组长和操作员）, 不可判定时为 None.
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
        # 实时落盘(不等到 save_results), 保证评估提取时 behavior_key_moments.json 有全部事件
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

    def _process_pop_view(
        self,
        video_path: str,
        video_source: str,
    ) -> None:
        """在 pop 视角上运行指向检测（YOLO 四分类 + MediaPipe）.

        Args:
            video_path: pop 视频文件路径.
            video_source: 推理流进度 source 名称.
        """
        detector = self._pointing_detector
        if detector is None:
            logger.error("[pop_view] 指向检测器未初始化，跳过推理")
            return

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
            total_duration = (total_frames / fps) if total_frames > 0 else 0.0
            frame_count = 0

            logger.info(
                f"[pop_view] 开始处理: {video_path} ({total_frames}帧, {fps:.1f}fps)"
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
                    current_time = time.time()
                    if current_time - self._last_progress_push >= 0.3:
                        self._last_progress_push = current_time
                        self.push_display(
                            "progress",
                            {
                                "localSec": round(timestamp, 2),
                                "tag": "progress",
                                "data": {
                                    "label": "pop",
                                    "pct": min(
                                        100.0,
                                        round(
                                            timestamp / total_duration * 100.0,
                                            1,
                                        ),
                                    ),
                                },
                            },
                        )

                # 先画 ROI 底层,检测器随后在帧上叠加检测框与手部关键点(in-place)
                draw_roi_overlay(frame, screen_polygons_as_tuples())
                events = detector.process_frame(frame, frame_count, fps)

                # POP 视角标注已由检测器画进帧, 此处把带标注帧喂入视觉流编码器 -> fMP4 -> 前端 MSE
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
                            f"未知行为事件类型: {event.get('event')}，跳过事件流推送"
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
                        f"[pop_view] {frame_count}/{total_frames}帧 {progress_percentage}%"
                    )

            video_capture.release()
            if self._vis_encoder is not None:
                self._vis_encoder.finalize()
            logger.info(
                f"[pop_view] 完成，共 {frame_count} 帧，{len(self._events)} 条事件"
            )

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

    def save_results(self, run_id: str) -> None:
        """收尾兜底保存: 实时落盘已逐条写入, 此处仅合并补写缺失事件(去重).

        Args:
            run_id: 本次运行标识.
        """
        with self._events_lock:
            events_list = list(self._events)
        self._result_storage.merge_events(run_id, events_list)
        if self._pointing_detector is not None:
            self._pointing_detector.close()
