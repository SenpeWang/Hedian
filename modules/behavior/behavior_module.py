"""
行为检测模块入口

继承 BaseModule，实现统一接口。
并行处理手指屏幕（camBUP）和手指文件（camPOP）两个视频，
统一保存所有行为事件到 behavior_key_moments.json。
"""
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2

from core.base_module import BaseModule
from core.event_bus import EventStream, EventTopic
from core.inference_stream import InferenceStream
from core.path_manager import PathManager

from modules.behavior.finger_bup_detector import FingerBupDetector
from modules.behavior.finger_pop_detector import FingerPopDetector
from modules.behavior.storage_behavior import BehaviorStorage

logger = logging.getLogger("module.behavior")


class BehaviorModule(BaseModule):
    """
    行为检测模块

    并行处理两个视角的行为检测：
    - 手指屏幕（camBUP.mpg）：FingerBupDetector
    - 手指文件（camPOP.mpg）：FingerPopDetector

    同时接收 tracker 推送的举手事件（BEHAVIOR_HAND_RAISED），统一保存。
    """

    def __init__(
        self,
        event_bus: EventStream,
        config: dict,
        paths: PathManager,
        inference_stream: InferenceStream,
    ):
        super().__init__(event_bus, config, paths, inference_stream)
        self._bup_detector = None
        self._pop_detector = None
        self._result_storage = None
        self._events = []
        self._events_lock = threading.Lock()

    @property
    def module_name(self) -> str:
        return "behavior"

    def initialize(self) -> bool:
        """初始化行为检测模块"""
        try:
            behavior_cfg = self.config.get("behavior", {})
            screen_cfg = behavior_cfg.get("finger_screen", {})
            file_cfg = behavior_cfg.get("finger_file", {})

            # 初始化手指屏幕检测器
            self._bup_detector = FingerBupDetector(
                pose_model_path=str(self.paths.get_model_path("behavior", "behavior_yolo26s-pose.pt")),
                finger_model_path=str(self.paths.get_model_path("behavior", "behavior_yolo.pt")),
                detect_conf=screen_cfg.get("detect_conf", 0.3),
                pose_conf=screen_cfg.get("pose_conf", 0.5),
                hand_to_screen_dist=screen_cfg.get("hand_to_screen_dist", 400),
                cooldown_sec=screen_cfg.get("cooldown_sec", 1.5),
            )

            # 初始化手指文件检测器
            self._pop_detector = FingerPopDetector(
                model_path=str(self.paths.get_model_path("behavior", "behavior_yolo.pt")),
                detect_conf=file_cfg.get("detect_conf", 0.25),
                track_iou=file_cfg.get("track_iou", 0.5),
                file_iou_threshold=file_cfg.get("file_iou_threshold", 0.2),
                cooldown_sec=file_cfg.get("cooldown_sec", 1.5),
            )

            # 初始化结果存储
            self._result_storage = BehaviorStorage(self.paths)

            # 订阅举手事件（由 tracker 嵌入检测后通过事件流推送）
            self.event_bus.subscribe(EventTopic.BEHAVIOR_HAND_RAISED, self._on_hand_raised)

            # 预声明 behavior source 归属（举手事件可能全程不触发，仍需在退出时上报结束信号，防卡 done）
            self._inference_sources.add("behavior")

            logger.info("行为检测模块初始化完成（手指屏幕 + 手指文件 + 举手订阅）")
            return True

        except Exception as e:
            logger.error(f"行为检测模块初始化失败: {e}", exc_info=True)
            return False

    def _on_hand_raised(self, msg: dict) -> None:
        """订阅 BEHAVIOR_HAND_RAISED：仅接收 tracker 在 front 视角检测到的真正举手事件"""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0))
        raw_op = data.get("operator") or msg.get("operator") or "ROAD1"
        operator = raw_op if (raw_op and raw_op != "UNKNOWN") else "ROAD1"
        with self._events_lock:
            self._events.append({
                "localSec": round(ts, 2),
                "key_moment": f"{operator}举手",
            })
        logger.info(f"行为模块收到举手事件: {operator} @{ts:.1f}s")

    # 视频帧推送间隔（每 N 帧推送一帧到前端）
    FRAME_PUSH_INTERVAL = 1

    def _process_single_video(
        self,
        video_path: str,
        detector,
        tag: str,
        video_source: str,
    ) -> None:
        """
        处理单个视频的通用流程

        Args:
            video_path: 视频文件路径
            detector: 检测器实例（FingerBupDetector 或 FingerPopDetector）
            tag: 推理事件标签（如 "behavior"）
            video_source: 视频帧推送源（如 "video_bup"）
        """
        name = detector.__class__.__name__

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"[{name}] 无法打开视频: {video_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or self.config.get("fps", 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            logger.info(f"[{name}] 开始处理: {video_path} ({total_frames}帧, {fps:.1f}fps)")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                ts = frame_count / fps
                frame_count += 1

                # 仅写本路视频自己的 source 进度（不调 update_progress 避免写全集导致两路串扰）
                self.inference_stream.update_module_time(video_source, ts)
                # 推送前端进度条（每秒最多一次，用 video_source 作 label 区分两路）
                if total_frames > 0:
                    now = time.time()
                    if not hasattr(self, "_last_progress_push"):
                        self._last_progress_push = 0
                    if now - self._last_progress_push >= 1.0:
                        self._last_progress_push = now
                        pct = min(100, ts / (total_frames / fps) * 100)
                        self.push_display("progress", {
                            "localSec": round(ts, 2),
                            "tag": "progress",
                            "data": {"label": video_source, "pct": round(pct, 1)},
                        })

                # 检测（同时在帧上绘制推理流标注）
                events = detector.detect(frame, frame_count, fps)

                # 推送带标注的视频帧到前端（每 N 帧一次）——推理流
                if frame_count % self.FRAME_PUSH_INTERVAL == 0:
                    frame_small = cv2.resize(frame, (960, 540))
                    ok, buf = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 40])
                    if ok:
                        self.push_display(video_source, {
                            "localSec": round(ts, 2),
                            "tag": "video",
                            "data": {"frame_data": buf.tobytes().decode("latin1")},
                        })

                # 事件流：关键事件既存储也推送
                for event in events:
                    event["localSec"] = round(ts, 2)
                    with self._events_lock:
                        self._events.append({
                            "localSec": event["localSec"],
                            "key_moment": event.get("state", tag),
                        })

                    # 推理流：前端展示
                    self.push_display(tag, {
                        "localSec": event["localSec"],
                        "tag": event.get("event", tag),
                        "data": {k: v for k, v in event.items() if k not in ("localSec",)},
                    })
                    # 事件流：后端模块间通信与规则状态机联动 (零兼容重构: 精准按动作类型分发 Topic，彻底清理旧 BEHAVIOR_HAND_RAISED)
                    evt_type = event.get("event", "")
                    if evt_type == "FINGER_FILE":
                        topic = EventTopic.BEHAVIOR_FINGER_FILE
                    elif evt_type == "FINGER_SCREEN":
                        topic = EventTopic.BEHAVIOR_FINGER_SCREEN
                    else:
                        topic = EventTopic.BEHAVIOR_HAND_RAISED

                    self.push_event(topic, {
                        "localSec": event["localSec"],
                        "tag": event.get("event", tag),
                        "data": {k: v for k, v in event.items() if k not in ("localSec",)},
                    }, ts=event["localSec"])

                # 进度日志
                if frame_count % 300 == 0:
                    pct = frame_count * 100 // total_frames if total_frames else 0
                    logger.info(f"[{name}] {frame_count}/{total_frames}帧 {pct}%")

            cap.release()
            logger.info(f"[{name}] 完成，共 {frame_count} 帧")

        except Exception as e:
            logger.error(f"[{name}] 视频处理失败: {e}", exc_info=True)

    def process_video(self, video_path: str) -> None:
        """
        并行处理两个视角的视频

        视频路径从 config 读取，忽略 BaseModule 传入的 video_path。
        """
        videos_cfg = self.config.get("videos", {})
        base = self.paths.base_dir
        bup_video = str(base / videos_cfg.get("bup", "data/videos/camBUP.mpg"))
        pop_video = str(base / videos_cfg.get("pop", "data/videos/camPOP.mpg"))

        logger.info("并行处理两个视频:")
        logger.info(f"  手指屏幕: {bup_video}")
        logger.info(f"  手指文件: {pop_video}")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(self._process_single_video, bup_video, self._bup_detector, "behavior", "video_bup"),
                executor.submit(self._process_single_video, pop_video, self._pop_detector, "behavior", "video_pop"),
            ]
            for f in futures:
                f.result()

        logger.info(f"两个视频处理完成，共 {len(self._events)} 条事件")

    def save_results(self, run_id: str) -> None:
        """保存行为检测结果（委托给 BehaviorStorage）"""
        with self._events_lock:
            # 保证按时间戳 localSec 升序排列
            sorted_events = sorted(self._events, key=lambda x: x.get("localSec", 0))
        self._result_storage.save_key_moments(run_id, sorted_events)
