"""
行为检测模块入口

继承 BaseModule，实现统一接口。
统一保存所有行为事件（举手、手指屏幕等）到 behavior_key_moments.json。
"""
import logging

from core.base_module import BaseModule
from core.event_bus import EventBus, EventTopic
from core.inference_bus import InferenceBus
from core.path_manager import PathConfig

from modules.behavior.finger_screen_detector import FingerScreenDetector
from modules.behavior.storage_behavior import BehaviorStorage

logger = logging.getLogger("module.behavior")


class BehaviorModule(BaseModule):
    """
    行为检测模块

    负责手指屏幕检测，并统一保存所有行为事件：
    - 举手事件（由 tracker 嵌入检测，通过 BEHAVIOR_HAND_RAISED 事件流推送过来）
    - 手指屏幕事件（本进程检测）
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: dict,
        paths: PathConfig,
        display_buffer: InferenceBus,
    ):
        super().__init__(event_bus, config, paths, display_buffer)
        self._detector = None
        self._result_storage = None
        self._events = []  # 统一保存所有行为事件（举手 + 手指屏幕 + ...）

    @property
    def module_name(self) -> str:
        return "behavior"

    def initialize(self) -> bool:
        """初始化行为检测模块"""
        try:
            # 初始化手指屏幕检测器
            self._detector = FingerScreenDetector(
                pose_model_path=self.paths.get_model_path("behavior", "yolo11l-pose.pt"),
                finger_model_path=self.paths.get_model_path("behavior", "yolov8_finger.pt"),
            )

            # 初始化结果存储
            self._result_storage = BehaviorStorage(self.paths)

            # 订阅举手事件（由 tracker 嵌入检测后通过事件流推送，本进程统一保存）
            self.event_bus.subscribe(EventTopic.BEHAVIOR_HAND_RAISED, self._on_hand_raised)

            logger.info("行为检测模块初始化完成（已订阅 BEHAVIOR_HAND_RAISED）")
            return True

        except Exception as e:
            logger.error(f"行为检测模块初始化失败: {e}", exc_info=True)
            return False

    def _on_hand_raised(self, msg: dict) -> None:
        """订阅 BEHAVIOR_HAND_RAISED：接收 tracker 检测到的举手事件，统一保存"""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0))
        operator = data.get("operator", "UNKNOWN")
        self._events.append({
            "localSec": round(ts, 2),
            "key_moment": f"{operator}举手",
        })
        logger.info(f"行为模块收到举手事件: {operator} @{ts:.1f}s")

    def process_video(self, video_path: str) -> None:
        """处理视频，进行行为检测（推理流统一格式：{source, localSec, tag, data}）"""
        import cv2

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or self.config.get("fps", 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            logger.info(f"开始处理视频: {video_path}")
            logger.info(f"FPS={fps}, 总帧数={total_frames}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                ts = frame_count / fps
                frame_count += 1

                # 更新进度
                self.update_progress(ts, total_frames / fps)

                # 检测手指屏幕
                events = self._detector.detect(frame, frame_count, fps)

                # 保存并推送事件
                for event in events:
                    event["localSec"] = round(ts, 2)
                    self._events.append({
                        "localSec": event["localSec"],
                        "key_moment": event.get("type", "behavior"),
                    })

                    # 推理流：前端展示行为检测
                    self.push_display("behavior", {
                        "localSec": event["localSec"],
                        "tag": event.get("type", "BEHAVIOR"),
                        "data": {k: v for k, v in event.items() if k not in ("localSec",)},
                    })

                # 进度日志
                if frame_count % 300 == 0:
                    pct = frame_count * 100 // total_frames if total_frames else 0
                    logger.info(f"{frame_count}/{total_frames}帧 {pct}%")

            cap.release()
            logger.info(f"视频处理完成，共 {frame_count} 帧")

        except Exception as e:
            logger.error(f"视频处理失败: {e}", exc_info=True)

    def save_results(self, run_id: str) -> None:
        """保存行为检测结果（委托给 BehaviorStorage）"""
        self._result_storage.save_key_moments(run_id, self._events)
