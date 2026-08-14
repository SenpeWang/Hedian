"""行为检测模块入口.

在单一 pop 视角（camPOP.mpg）上共享一次 YOLO 推理，
串行运行手指指向屏幕（FingerScreenDetector）与手指指向文件（FingerFileDetector）两个判定器，
举手检测由 tracker 模块在前置视角调用 HandRaiser，本模块通过 BehaviorStorage 共享事件落盘。
"""
import logging
import threading
import time
from typing import List, Dict, Optional, Any

import cv2
from ultralytics import YOLO

from core.base_module import BaseModule
from core.event_bus import EventStream, EventTopic
from core.inference_stream import InferenceStream
from core.path_manager import PathManager

from modules.behavior.screen_detect import FingerScreenDetector
from modules.behavior.file_detector import FingerFileDetector
from modules.behavior.storage_behavior import BehaviorStorage

logger = logging.getLogger("module.behavior")

_EVENT_TOPIC_MAP: Dict[str, EventTopic] = {
    "FINGER_SCREEN": EventTopic.BEHAVIOR_FINGER_SCREEN,
    "FINGER_FILE": EventTopic.BEHAVIOR_FINGER_FILE,
}


class BehaviorModule(BaseModule):
    """行为检测模块.

    管理行为事件（手指屏幕、手指文件、举手）的推理、推送与结果落盘。
    """

    def __init__(
        self,
        event_bus: EventStream,
        config: dict,
        paths: PathManager,
        inference_stream: InferenceStream,
    ):
        """初始化行为检测模块.

        Args:
            event_bus (EventStream): 全局事件总线.
            config (dict): 全局配置字典.
            paths (PathManager): 路径管理器.
            inference_stream (InferenceStream): 前端推理流推送通道.
        """
        super().__init__(event_bus, config, paths, inference_stream)
        self._screen_detector: Optional[FingerScreenDetector] = None
        self._file_detector: Optional[FingerFileDetector] = None
        self._result_storage: Optional[BehaviorStorage] = None
        self._events: List[Dict[str, Any]] = []
        self._events_lock = threading.Lock()
        self._last_progress_push: float = 0.0

    @property
    def module_name(self) -> str:
        """获取模块名称.

        Returns:
            str: 模块标识字符串 'behavior'.
        """
        return "behavior"

    def initialize(self) -> bool:
        """初始化行为检测模块的各判定器与事件订阅.

        Returns:
            bool: 初始化成功返回 True，失败返回 False.
        """
        try:
            cfg = self.config.get("behavior", {})
            screen_cfg = cfg.get("screen", {})
            file_cfg = cfg.get("file", {})

            fps = self._read_video_fps("pop", 30.0)

            self._screen_detector = FingerScreenDetector(
                detect_conf=screen_cfg.get("detect_conf", 0.25),
                screen_overlap_threshold=screen_cfg.get("screen_overlap_threshold", 0.2),
                max_dist=screen_cfg.get("max_dist", 20),
                cooldown_sec=screen_cfg.get("cooldown_sec", 1.5),
                fps=fps,
            )

            self._file_detector = FingerFileDetector(
                detect_conf=file_cfg.get("detect_conf", 0.25),
                file_iou_threshold=file_cfg.get("file_iou_threshold", 0.2),
                cooldown_sec=file_cfg.get("cooldown_sec", 1.5),
                fps=fps,
            )

            self._result_storage = BehaviorStorage(self.paths)
            self.event_bus.subscribe(EventTopic.BEHAVIOR_HAND_RAISED, self._on_hand_raised)

            logger.info("行为检测模块初始化完成")
            return True

        except Exception as init_error:
            logger.error(f"行为检测模块初始化失败: {init_error}", exc_info=True)
            return False

    def _read_video_fps(self, video_key: str, default_fps: float = 30.0) -> float:
        """读取配置中指定视角视频的真实帧率.

        Args:
            video_key (str): 视频键名（如 'pop'）.
            default_fps (float): 读取失败时的回退默认帧率.

        Returns:
            float: 视频真实帧率或默认帧率.
        """
        try:
            videos_cfg = self.config.get("videos", {})
            rel_path = videos_cfg.get(video_key, f"data/videos/cam{video_key.upper()}.mpg")
            abs_path = str(self.paths.base_dir / rel_path)
            cap = cv2.VideoCapture(abs_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps and fps > 0:
                    return float(fps)
        except Exception as init_error:
            logger.warning(f"读取视频帧率失败 ({video_key}): {init_error}，使用默认值 {default_fps}")
        return default_fps

    def _append_event(self, local_sec: float, key_moment: str) -> None:
        """线程安全地收集关键事件.

        Args:
            local_sec (float): 事件发生的时间戳（秒）.
            key_moment (str): 关键时刻描述文本.
        """
        with self._events_lock:
            self._events.append({
                "localSec": round(local_sec, 2),
                "key_moment": key_moment,
            })

    def _on_hand_raised(self, msg: dict) -> None:
        """订阅 BEHAVIOR_HAND_RAISED 事件的回调函数.

        Args:
            msg (dict): 事件总线消息字典.
        """
        data = msg.get("data", {})
        operator_name = data.get("operator") or msg.get("operator")
        timestamp = data.get("localSec", msg.get("ts", 0))
        logger.info(f"行为模块收到举手事件: {operator_name} @{timestamp:.1f}s")

    FRAME_PUSH_INTERVAL: int = 1

    def _process_pop_view(
        self,
        video_path: str,
        model: Any,
        screen_detector: FingerScreenDetector,
        file_detector: FingerFileDetector,
        tag: str,
        video_source: str,
    ) -> None:
        """在单一 pop 视角上共享一次推理并运行两个判定器.

        Args:
            video_path (str): pop 视频文件路径.
            model (Any): 共享的 YOLO 行为检测模型实例.
            screen_detector (FingerScreenDetector): 手指屏幕检测器.
            file_detector (FingerFileDetector): 手指文件检测器.
            tag (str): 推理事件标签.
            video_source (str): 视频流推送源名称.
        """
        try:
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                logger.error(f"[pop_view] 无法打开视频: {video_path}")
                return

            fps = video_capture.get(cv2.CAP_PROP_FPS) or self.config.get("fps", 30.0)
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            infer_every_n_frames = 5
            cached_results = None

            logger.info(f"[pop_view] 开始处理: {video_path} ({total_frames}帧, {fps:.1f}fps)")

            while True:
                frame_read_success, frame = video_capture.read()
                if not frame_read_success:
                    break

                timestamp = frame_count / fps
                frame_count += 1

                self.inference_stream.update_module_time(video_source, timestamp)
                if total_frames > 0:
                    current_time = time.time()
                    if current_time - self._last_progress_push >= 1.0:
                        self._last_progress_push = current_time
                        progress_percentage = min(100.0, timestamp / (total_frames / fps) * 100)
                        self.push_display("progress", {
                            "localSec": round(timestamp, 2),
                            "tag": "progress",
                            "data": {"label": video_source, "pct": round(progress_percentage, 1)},
                        })

                if frame_count % infer_every_n_frames == 0 or cached_results is None:
                    cached_results = model.track(
                        frame,
                        conf=0.2,
                        iou=0.5,
                        persist=True,
                        verbose=False,
                    )[0]
                results = cached_results

                events = []
                events += screen_detector.detect(frame, results, frame_count, fps)
                events += file_detector.detect(frame, results, frame_count, fps)

                if frame_count % self.FRAME_PUSH_INTERVAL == 0:
                    frame_small = cv2.resize(frame, (960, 540))
                    encode_success, jpeg_buffer = cv2.imencode(
                        ".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 40]
                    )
                    if encode_success:
                        self.push_display(video_source, {
                            "localSec": round(timestamp, 2),
                            "tag": "video",
                            "data": {"frame_data": jpeg_buffer.tobytes().decode("latin1")},
                        })

                for event_item in events:
                    event_sec = round(event_item.get("localSec", timestamp), 2)
                    self._append_event(event_sec, event_item.get("state", tag))

                    payload = {
                        "localSec": event_sec,
                        "tag": event_item.get("event", tag),
                        "data": {
                            dict_key: dict_value
                            for dict_key, dict_value in event_item.items()
                            if dict_key != "localSec"
                        },
                    }
                    self.push_display(tag, payload)
                    topic = _EVENT_TOPIC_MAP.get(event_item.get("event", ""))
                    if topic is None:
                        logger.warning(f"未知行为事件类型: {event_item.get('event')}，跳过事件流推送")
                        continue
                    self.push_event(topic, payload, ts=event_sec)

                if frame_count % 300 == 0:
                    progress_pct = frame_count * 100 // total_frames if total_frames else 0
                    logger.info(f"[pop_view] {frame_count}/{total_frames}帧 {progress_pct}%")

            video_capture.release()
            logger.info(f"[pop_view] 完成，共 {frame_count} 帧")

        except Exception as pop_error:
            logger.error(f"[pop_view] 视频处理失败: {pop_error}", exc_info=True)

    def process_video(self, video_path: str) -> None:
        """在单一 pop 视角上处理行为检测.

        Args:
            video_path (str): 视频文件路径（可被配置项覆盖）.
        """
        videos_cfg = self.config.get("videos", {})
        base = self.paths.base_dir
        pop_video = str(base / videos_cfg.get("pop", "data/videos/camPOP.mpg"))

        model_path = str(self.paths.get_model_path("behavior", "behavior_yolo.pt"))
        model = YOLO(model_path)

        logger.info("在 pop 视角上共享一次推理运行两个行为检测器:")
        logger.info(f"  手指屏幕 + 手指文件: {pop_video}")

        self._process_pop_view(
            pop_video,
            model,
            self._screen_detector,
            self._file_detector,
            "behavior",
            "video_pop",
        )

        logger.info(f"pop 视角处理完成，共 {len(self._events)} 条事件")

    def save_results(self, run_id: str) -> None:
        """保存行为检测结果到 behavior_key_moments.json.

        Args:
            run_id (str): 本次运行标识.
        """
        with self._events_lock:
            events_list = list(self._events)
        for event_item in events_list:
            self._result_storage.add_event(run_id, event_item)
