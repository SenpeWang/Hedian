"""
行为检测模块入口

继承 BaseModule，实现统一接口。
"""
import os
import json
import logging

from core.base_module import BaseModule
from core.message_bus import MessageBus, MsgType
from core.frontend_sync import FrontendSync
from core.path_manager import PathConfig

from modules.behavior.finger_screen_detector import FingerScreenDetector

logger = logging.getLogger("module.behavior")


class BehaviorModule(BaseModule):
    """
    行为检测模块

    负责手指屏幕检测。
    """

    def __init__(
        self,
        bus: MessageBus,
        config: dict,
        paths: PathConfig,
        aggregator: FrontendSync,
    ):
        super().__init__(bus, config, paths, aggregator)
        self._detector = None
        self._events = []

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

            logger.info("行为检测模块初始化完成")
            return True

        except Exception as e:
            logger.error(f"行为检测模块初始化失败: {e}", exc_info=True)
            return False

    def process_video(self, video_path: str) -> None:
        """处理视频，进行行为检测"""
        import cv2
        import queue as queue_mod
        import numpy as np

        # 获取视频帧队列（支持 Redis 和内存两种模式）
        frame_queue = self.config.get("_frame_queue", None)
        if frame_queue is None:
            redis_key = self.config.get("_frame_queue_redis_key", None)
            if redis_key:
                from core.redis_queue import RedisQueue
                frame_queue = RedisQueue(
                    key=redis_key,
                    redis_host=self.config.get("_redis_host", "localhost"),
                    redis_port=self.config.get("_redis_port", 6379),
                    redis_db=self.config.get("_redis_db", 0),
                    maxsize=8,
                )

        # 生成占位图（等待模块开发）
        if frame_queue is not None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder[:] = (20, 20, 30)  # 深色背景
            # 文字
            cv2.putText(placeholder, "Behavior Detection", (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 212, 255), 2)
            cv2.putText(placeholder, "Waiting for module...", (140, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 120, 144), 2)
            cv2.putText(placeholder, "No camera angle available", (110, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 100, 120), 2)
            # 边框
            cv2.rectangle(placeholder, (10, 10), (630, 470), (0, 212, 255), 2)
            _, jpeg = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 70])
            # 持续推送占位图
            import time
            while self._running:
                try:
                    frame_queue.put_nowait(jpeg.tobytes())
                except Exception:
                    try:
                        frame_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        frame_queue.put_nowait(jpeg.tobytes())
                    except Exception:
                        pass
                time.sleep(0.1)  # 10fps

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

                # 保存事件
                for event in events:
                    event["localSec"] = round(ts, 2)
                    self._events.append(event)

                    # 发布到总线
                    self.bus.publish(MsgType.BEHAVIOR_FINGER_SCREEN, event, ts=ts)

                    # 推送到聚合器
                    self.push_event("behavior", event)

                # 进度日志
                if frame_count % 300 == 0:
                    pct = frame_count * 100 // total_frames if total_frames else 0
                    logger.info(f"{frame_count}/{total_frames}帧 {pct}%")

                # 可视化帧（每10帧）
                if frame_queue is not None and frame_count % 10 == 0:
                    try:
                        vis_frame = frame.copy()
                        # 画检测结果
                        if events:
                            for event in events:
                                ev_type = event.get("type", "")
                                if "finger" in ev_type.lower():
                                    cv2.putText(vis_frame, f"FINGER: {ev_type}",
                                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.7, (0, 0, 255), 2)
                        _, jpeg = cv2.imencode(".jpg", vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        try:
                            frame_queue.put_nowait(jpeg.tobytes())
                        except Exception:
                            try:
                                frame_queue.get_nowait()
                            except Exception:
                                pass
                            frame_queue.put_nowait(jpeg.tobytes())
                    except Exception as e:
                        logger.warning(f"可视化帧失败: {e}")

            cap.release()
            logger.info(f"视频处理完成，共 {frame_count} 帧")

        except Exception as e:
            logger.error(f"视频处理失败: {e}", exc_info=True)

    def save_results(self, run_id: str) -> None:
        """保存行为检测结果"""
        if not self._events:
            logger.warning("没有事件可保存")
            return

        # 保存关键事件
        output_path = self.paths.get_result_path(
            run_id=run_id,
            module="behavior",
            filename="Behavior_key_moments.json",
        )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self._events, f, ensure_ascii=False, indent=2)

            logger.info(f"保存 {len(self._events)} 个关键事件到 {output_path}")

        except Exception as e:
            logger.error(f"保存关键事件失败: {e}", exc_info=True)
