"""多目标跟踪模块入口.

继承 BaseModule，实现多目标检测、跟踪、前置摄像头举手识别调用及距离人数监控。
"""
import logging
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from core.base_module import BaseModule
from core.event_bus import EventStream, EventTopic
from core.inference_stream import InferenceStream
from core.path_manager import PathManager

from modules.tracker.object_detector import ObjectDetector
from modules.tracker.multi_object_tracker import MultiObjectTracker, STrack
from modules.behavior.hand_raiser import HandRaiser
from modules.behavior.storage_behavior import BehaviorStorage
from modules.tracker.storage_tracker import TrackerStorage
from modules.tracker.visualizer import draw_tracks
from modules.gaze.gaze_module import GazeModule

logger = logging.getLogger("module.tracker")


class TrackerModule(BaseModule):
    """多目标跟踪模块.

    负责主控室人员检测、跟踪、调用行为模块举手识别、监控人员工位距离与在岗人数。
    """

    def __init__(
        self,
        event_bus: EventStream,
        config: dict,
        paths: PathManager,
        inference_stream: InferenceStream,
    ):
        """初始化多目标跟踪模块.

        Args:
            event_bus (EventStream): 全局事件总线.
            config (dict): 全局配置字典.
            paths (PathManager): 路径管理器.
            inference_stream (InferenceStream): 前端推理流推送通道.
        """
        super().__init__(event_bus, config, paths, inference_stream)
        self._detector: Optional[ObjectDetector] = None
        self._tracker: Optional[MultiObjectTracker] = None
        self._hand_raiser: Optional[HandRaiser] = None
        self._gaze_processor: Optional[GazeModule] = None
        self._result_storage: Optional[TrackerStorage] = None
        self._events: List[Dict] = []
        self._identity_map: Dict[str, int] = {}
        self._last_supervision_states: Dict[str, str] = {}
        self._supervision_active: bool = False

    @property
    def module_name(self) -> str:
        """模块名称."""
        return "tracker"

    def initialize(self) -> bool:
        """初始化 MOT 模块各子组件与事件订阅.

        Returns:
            bool: 初始化成功返回 True，失败返回 False.
        """
        try:
            tracker_config = self.config.get("tracker", {})
            supervision_config = self.config.get("supervision", {})

            # 初始化目标与姿态检测器
            detection_config = tracker_config.get("detection", {})
            self._detector = ObjectDetector(
                model_path=self.paths.get_model_path("detection", "yolo11_MOT.pt"),
                pose_model_path=self.paths.get_model_path("detection", "yolo26s-pose.pt"),
                conf_threshold=detection_config.get("conf_threshold", 0.65),
                pose_confidence=detection_config.get("pose_confidence", 0.3),
                nms_threshold=detection_config.get("nms_threshold", 0.35),
                img_size=detection_config.get("img_size", 640),
            )

            # 初始化多目标跟踪器
            self._tracker = MultiObjectTracker()

            # 初始化凝视处理器
            gaze_config = self.config.get("gaze", {})
            self._gaze_processor = GazeModule(
                head_model_path=self.paths.get_model_path("gaze", "yolov8n_head.onnx"),
                gaze_model_path=self.paths.get_model_path("gaze", "gazelle_dinov3_vits16plus_finetuned_1x3x640x640_1xNx4.onnx"),
                roi_json_path=str(self.paths.base_dir / "data" / "ROI.json"),
                config=gaze_config,
                inference_fn=self.push_display,
                event_bus=self.event_bus,
                progress_fn=lambda cur, total: self.inference_stream.update_module_time("gaze", cur),
                paths=self.paths,
            )

            # 初始化举手检测器（实现位于 behavior 模块，tracker 仅持有实例用于前置视角调用）
            cooldown_sec = float(supervision_config.get("cooldown_sec", 1.5))
            fps = float(self.config.get("fps", 30.0))
            self._hand_raiser = HandRaiser(
                detector=self._detector,
                consec_raise=int(supervision_config.get("consec_raise", 2)),
                consec_idle=int(supervision_config.get("consec_idle", 3)),
                cooldown_frames=int(fps * cooldown_sec),
            )

            # 初始化结果存储管理器
            self._result_storage = TrackerStorage(self.paths)

            # 订阅流程与规则事件
            self.event_bus.subscribe(EventTopic.FLOW_STARTED, self._on_flow_started)
            self.event_bus.subscribe(EventTopic.FLOW_ENDED, self._on_flow_ended)
            self.event_bus.subscribe(EventTopic.RULE_KEY_MOMENT, self._on_rule_key_moment)

            # behavior source（举手事件）由 tracker 代推但归属 behavior 模块，退出时不标记结束
            self._borrowed_sources = {"behavior"}
            # gaze 内嵌 tracker 但进度由 GazeModule 异步独立写，update_progress 时跳过
            self._independent_progress_sources = {"gaze"}

            logger.info("MOT 跟踪模块初始化完成")
            return True

        except Exception as init_error:
            logger.error(f"MOT 模块初始化失败: {init_error}", exc_info=True)
            return False

    def process_video(self, video_path: str) -> None:
        """处理前置监控视频流，执行人员检测、跟踪、行为调用与状态监控.

        Args:
            video_path (str): 视频文件路径.
        """
        try:
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return

            fps = video_capture.get(cv2.CAP_PROP_FPS) or self.config.get("fps", 30.0)
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            # 确保关键帧目录已准备就绪
            self._result_storage.get_key_frames_dir(self._run_id)

            identities_assigned_done = False

            logger.info(f"开始处理视频: {video_path}")
            logger.info(f"FPS={fps}, 总帧数={total_frames}")

            # 通知前端启动视频流
            self.push_display("video_start", {"localSec": 0, "tag": "start", "data": {}})

            tracks: List[STrack] = []

            while True:
                frame_read_success, frame = video_capture.read()
                if not frame_read_success:
                    break

                timestamp = frame_count / fps
                frame_count += 1

                # 更新进度
                self.update_progress(timestamp, total_frames / fps)

                # 更新聚合器快照
                self.inference_stream.update_module_snapshot("tracker", {
                    "roles": dict(self._identity_map),
                    "localSec": round(timestamp, 2),
                })

                # 1. 目标检测
                high_conf_dets, low_conf_dets = self._detector.detect_two_thresholds(frame)
                detections = high_conf_dets + low_conf_dets

                # 2. 目标跟踪
                tracks = self._tracker.track(frame, detections)

                # 3. 工位身份分配
                if not self._tracker.identities_assigned and self._tracker.initialized and len(tracks) >= 2:
                    self._tracker.assign_identities(tracks)

                # 身份分配事件广播与落盘
                if self._tracker.identities_assigned and not identities_assigned_done:
                    identities_assigned_done = True
                    self._identity_map = dict(self._tracker.identity_map)

                    roles_formatted_str = ",".join(
                        f"{role_name}:{track_id}"
                        for role_name, track_id in self._identity_map.items()
                    )
                    self._events.append({
                        "localSec": round(timestamp, 2),
                        "key_moment": f"角色分配,{roles_formatted_str}",
                        "source": "tracker",
                    })
                    if self._result_storage:
                        self._result_storage.save_key_moments(self._run_id, self._events)
                        self._result_storage.save_role_info(self._run_id, self._identity_map)

                    # 推理流：前端展示身份分配结果
                    self.push_display("tracking", {
                        "localSec": round(timestamp, 2),
                        "tag": "ROLE_ASSIGNED",
                        "data": {"roles": self._identity_map}
                    })

                    # 保存身份分配关键帧
                    self._result_storage.save_key_frame(
                        self._run_id, f"role_assigned_{timestamp:.1f}s.jpg", frame
                    )
                    logger.info(f"工位身份分配完成: {self._identity_map}")

                # 4. 举手检测：对全量跟踪目标进行判别（实现位于 behavior 模块，署名全归属 behavior）
                raised_targets = self._hand_raiser.check(
                    frame, tracks, frame_count
                )
                for track_id, identity in raised_targets:
                    BehaviorStorage(self.paths).report_hand_raise(
                        self.event_bus,
                        self._run_id,
                        identity=identity,
                        timestamp=timestamp,
                        track_id=track_id,
                        inference_fn=self.push_display,
                    )

                    # 保存关键帧（由 TrackerStorage 统管写入）
                    label = identity if identity else f"track_{track_id}"
                    self._result_storage.save_key_frame(
                        self._run_id, f"hand_raise_{label}_{timestamp:.1f}s.jpg", frame
                    )

                    tag_label = f"身份={identity}" if identity else "未分配身份"
                    logger.info(
                        f"举手检测: {tag_label} (track_id={track_id}) @{timestamp:.1f}s"
                    )

                # 5. 距离与人数监控（每 30 帧检查一次）
                if frame_count % 30 == 0:
                    self._monitor_distance(timestamp)
                    self._monitor_headcount(timestamp, tracks)

                # 6. 可视化绘制与推理流逐帧全量推送
                try:
                    annotated_vis_frame = draw_tracks(frame, tracks, self._identity_map)
                    if self._tracker.identities_assigned:
                        self._draw_distance_lines(annotated_vis_frame)
                    annotated_vis_frame = self._draw_gaze(
                        annotated_vis_frame, frame_count, timestamp, tracks
                    )

                    # 降分辨率到 720p 并压缩为 JPEG 发送
                    vis_small = cv2.resize(annotated_vis_frame, (960, 540))
                    encode_success, jpeg_buffer = cv2.imencode(
                        ".jpg", vis_small, [cv2.IMWRITE_JPEG_QUALITY, 35]
                    )
                    if encode_success:
                        self.push_display("video_front", {
                            "localSec": round(timestamp, 2),
                            "tag": "frame",
                            "data": {
                                "frame_data": jpeg_buffer.tobytes().decode('latin1'),
                                "frame_id": frame_count,
                            },
                        })
                except Exception as vis_error:
                    logger.warning(f"可视化帧失败: {vis_error}")

                # 进度日志
                if frame_count % 300 == 0:
                    percentage = frame_count * 100 // total_frames if total_frames else 0
                    logger.info(f"{frame_count}/{total_frames}帧 {percentage}%")

            video_capture.release()
            logger.info(f"视频处理完成，共 {frame_count} 帧")

        except Exception as process_error:
            logger.error(f"视频处理失败: {process_error}", exc_info=True)

    def save_results(self, run_id: str) -> None:
        """保存 MOT 结果（委托给 TrackerStorage + gaze）."""
        self._result_storage.save_key_moments(run_id, self._events)
        self._result_storage.save_role_info(run_id, self._identity_map)
        self._gaze_processor.save_results(run_id)
        logger.info(f"MOT 结果已保存到 {run_id}")

    def _on_flow_started(self, msg: dict) -> None:
        """订阅 FLOW_STARTED：监护制流程开始时触发."""
        data = msg.get("data", {})
        if data.get("flow_type") == "supervision":
            self._supervision_active = True
            flow_start_timestamp = data.get("flow_start_sec", 0)
            logger.info(f"Tracker: 监护制流程开始 @{flow_start_timestamp:.1f}s")

    def _on_flow_ended(self, msg: dict) -> None:
        """订阅 FLOW_ENDED：流程结束时触发."""
        data = msg.get("data", {})
        if data.get("flow_type") == "supervision":
            self._supervision_active = False
            flow_end_timestamp = data.get("flow_end_sec", 0)
            logger.info(f"Tracker: 监护制流程结束 @{flow_end_timestamp:.1f}s")

    def _on_rule_key_moment(self, msg: dict) -> None:
        """订阅 RULE_KEY_MOMENT：规则层下发的监护绑定/解绑 key_moment 归属 tracker."""
        data = msg.get("data", {})
        timestamp = data.get("localSec", msg.get("ts", 0))
        key_moment = data.get("key_moment", "")
        if not key_moment:
            return
        self._events.append({
            "localSec": round(timestamp, 2),
            "key_moment": key_moment,
            "source": "tracker",
        })
        if self._result_storage:
            self._result_storage.save_key_moments(self._run_id, self._events)
        logger.debug(f"Tracker 接收 RULE_KEY_MOMENT: {key_moment} @{timestamp:.1f}s")

    def update_progress(self, current: float, total: float = None) -> None:
        """更新 tracker 进度并推送 gaze 前端进度条."""
        super().update_progress(current, total)
        if total and total > 0:
            current_time = time.time()
            if not hasattr(self, "_last_gaze_progress_push"):
                self._last_gaze_progress_push = 0
            if current_time - self._last_gaze_progress_push >= 1.0:
                self._last_gaze_progress_push = current_time
                gaze_percentage = min(100.0, current / total * 100)
                self.push_display("progress", {
                    "localSec": round(current, 2),
                    "tag": "progress",
                    "data": {"label": "gaze", "pct": round(gaze_percentage, 1)},
                })

    def _monitor_distance(self, timestamp: float) -> None:
        """距离监控：判定 LEADER 与 ROAD1/ROAD2 距离状态并推送."""
        if not self._tracker.identities_assigned:
            return
        leader = self._tracker.get_track_by_identity("LEADER")
        if not leader:
            return
        for road_name in ("ROAD1", "ROAD2"):
            target = self._tracker.get_track_by_identity(road_name)
            if not target:
                continue
            distance = float(np.linalg.norm(leader.get_center() - target.get_center()))
            dist_close = self.config.get("supervision", {}).get("dist_close_px", 200)
            dist_near = self.config.get("supervision", {}).get("dist_near_px", 560)

            if distance <= dist_close:
                state_label = "监护中"
            elif distance <= dist_near:
                state_label = "接近中"
            else:
                state_label = "未监护"

            last_state = self._last_supervision_states.get(road_name)
            is_state_changed = (state_label != last_state)
            self._last_supervision_states[road_name] = state_label

            event_data = {
                "localSec": round(timestamp, 2),
                "tag": "SUPERVISOR_STATUS",
                "data": {"state": state_label, "operator": road_name, "distance_px": int(distance)},
            }

            if self._supervision_active:
                self.push_display("tracking", event_data)

            if self._supervision_active or is_state_changed:
                self.push_event(EventTopic.TRACKER_PROXIMITY, {
                    "localSec": round(timestamp, 2),
                    "state": state_label,
                    "operator": road_name,
                    "distance_px": int(distance),
                }, ts=timestamp)

    def _monitor_headcount(self, timestamp: float, tracks: list) -> None:
        """人数监控：人数变化时推送，无人值守时保存 key_moment."""
        people_count = len(tracks)
        last_count = getattr(self, "_last_people_count", -1)
        if people_count == last_count:
            return
        self._last_people_count = people_count

        people_data = {
            "localSec": round(timestamp, 2),
            "tag": "PEOPLE_COUNT_UPDATE",
            "data": {"count": people_count},
        }

        if people_count == 1:
            people_data["data"]["state"] = "主控室仅有1人"
            people_data["data"]["state_alert"] = "提醒：当前主控室只有一人，请注意安全！"
            self.push_display("tracking", people_data)
        elif people_count < 1:
            people_data["data"]["state"] = "主控室无人值守"
            people_data["data"]["state_alert"] = "警告：当前主控室无人值守！"
            self.push_display("tracking", people_data)
        else:
            self.push_display("tracking", people_data)

        if people_count < 1:
            self._events.append({
                "localSec": round(timestamp, 2),
                "key_moment": "主监控室少于1人",
                "source": "tracker",
            })
            if self._result_storage:
                self._result_storage.save_key_moments(self._run_id, self._events)
            self.push_event(EventTopic.TRACKER_HEADCOUNT, {
                "localSec": round(timestamp, 2),
                "count": people_count,
            }, ts=timestamp)

    def _draw_distance_lines(self, vis_frame: np.ndarray) -> None:
        """在可视化帧上绘制 LEADER 与 ROAD1/ROAD2 的距离线."""
        leader = self._tracker.get_track_by_identity("LEADER")
        if not leader:
            return
        for road_name in ("ROAD1", "ROAD2"):
            target = self._tracker.get_track_by_identity(road_name)
            if not target:
                continue
            leader_center = leader.get_center()
            road_center = target.get_center()
            distance = float(np.linalg.norm(leader_center - road_center))
            cv2.line(
                vis_frame,
                tuple(map(int, leader_center)),
                tuple(map(int, road_center)),
                (128, 128, 128),
                2,
            )
            mid_point = (
                (int(leader_center[0]) + int(road_center[0])) // 2,
                (int(leader_center[1]) + int(road_center[1])) // 2,
            )
            cv2.putText(
                vis_frame,
                f"{road_name} {int(distance)}px",
                mid_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )

    def _draw_gaze(
        self,
        frame: np.ndarray,
        frame_count: int,
        timestamp: float,
        tracks: Optional[list] = None,
    ) -> np.ndarray:
        """调用凝视处理器：异步检测、绘制缓存结果、推送."""
        return self._gaze_processor.process_frame(frame, timestamp, frame_count)
