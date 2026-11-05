"""多目标跟踪模块入口.

继承 BaseModule，实现多目标检测、跟踪、前置摄像头举手识别调用及距离人数监控。
"""

import logging
from pathlib import Path
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from core.base_module import BaseModule
from core.event_bus import EventBus, EventTopic
from core.inference_stream import InferenceStream
from core.path_manager import PathConfig
from core.vis_encoder import VisEncoder

from modules.behavior.hand_raiser import HandRaiser
from modules.behavior.storage_behavior import BehaviorStorage
from modules.gaze.gaze_module import GazeModule
from modules.identity.identity_monitor import IdentityMonitor
from modules.identity.storage_identity import IdentityStorage
from modules.tracker.multi_object_tracker import MultiObjectTracker, STrack
from modules.tracker.object_detector import ObjectDetector
from modules.tracker.scene_config import SceneConfig, load_scene
from modules.tracker.storage_tracker import TrackerStorage
from modules.tracker.visualizer import draw_tracks

logger = logging.getLogger("module.tracker")

# 视频元数据可接受的帧率上限(超过即视为畸形时间戳, 回退配置帧率)
MAX_REASONABLE_FPS = 240.0

# 默认模型权重文件名(相对 model_root/<子目录>, 可被 config.yaml 覆盖)
_DEFAULT_DET_WEIGHTS = "yolo11_MOT.pt"
_DEFAULT_DET_POSE = "yolo26s-pose.pt"
_DEFAULT_HEAD_MODEL = "yolov8n_head.onnx"
_DEFAULT_GAZE_MODEL = (
    "gazelle_dinov3_vits16plus_finetuned_1x3x640x640_1xNx4.onnx"
)


class TrackerModule(BaseModule):
    """多目标跟踪模块.

    负责主控室人员检测、跟踪、调用行为模块举手识别、监控人员工位距离与在岗人数。
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: dict,
        paths: PathConfig,
        inference_stream: InferenceStream,
    ) -> None:
        """初始化多目标跟踪模块.

        Args:
            event_bus: 全局事件总线.
            config: 全局配置字典.
            paths: 路径管理器.
            inference_stream: 前端推理流推送通道.
        """
        super().__init__(event_bus, config, paths, inference_stream)
        self._detector: Optional[ObjectDetector] = None
        self._tracker: Optional[MultiObjectTracker] = None
        self._hand_raiser: Optional[HandRaiser] = None
        self._gaze_processor: Optional[GazeModule] = None
        self._result_storage: Optional[TrackerStorage] = None
        self._events: List[Dict] = []
        self._identity_map: Dict[str, int] = {}
        self._scene: Optional[SceneConfig] = None
        self._identity_monitor: Optional[IdentityMonitor] = None
        self._identity_storage: Optional[IdentityStorage] = None
        self._vis_encoder: Optional[VisEncoder] = None
        self._last_supervision_states: Dict[str, str] = {}
        self._supervision_active: bool = False

    @property
    def module_name(self) -> str:
        """模块名称.

        Returns:
            模块名称字符串.
        """
        return "tracker"

    def initialize(self) -> bool:
        """初始化 MOT 模块各子组件与事件订阅.

        Returns:
            初始化成功返回 True，失败返回 False.
        """
        try:
            tracker_config = self.config.get("tracker", {})
            supervision_config = self.config.get("supervision", {})

            # 场景(机室)配置：工位基准点、判定多边形、屏幕 ROI 全部按机室加载
            self._scene = load_scene(self.config, self.paths.base_dir)

            # 初始化目标与姿态检测器
            detection_config = tracker_config.get("detection", {})
            self._detector = ObjectDetector(
                model_path=self.paths.get_model_path(
                    "detection",
                    detection_config.get("weights", _DEFAULT_DET_WEIGHTS),
                ),
                pose_model_path=self.paths.get_model_path(
                    "detection",
                    detection_config.get("pose_model", _DEFAULT_DET_POSE),
                ),
                conf_threshold=detection_config.get("conf_threshold", 0.65),
                pose_confidence=detection_config.get("pose_confidence", 0.3),
                nms_threshold=detection_config.get("nms_threshold", 0.35),
                img_size=detection_config.get("img_size", 640),
            )

            # 初始化多目标跟踪器（工位基准点按当前机室注入）
            self._tracker = MultiObjectTracker(
                workstations=self._scene.anchors or None
            )

            # 身份监控内嵌：复用本进程的 tracks，避免与独立进程各跑一遍 YOLO
            # 开关只看 identity.embedded：modules.identity 仅决定是否另起独立进程，
            # 不再作为内嵌的前置条件（否则两个开关不一致时会静默不跑身份监控）
            identity_config = self.config.get("identity", {}) or {}
            if identity_config.get("embedded", False):
                if not self.config.get("modules", {}).get("identity", False):
                    logger.warning(
                        "identity.embedded=true 但 modules.identity=false："
                        "身份监控仍由 tracker 进程内嵌承载，只是不起独立进程"
                    )
                self._identity_monitor = IdentityMonitor(
                    self._scene,
                    face_config=identity_config.get("face", {}) or {},
                    lock_config=identity_config.get("lock", {}) or {},
                    away_hold_sec=float(identity_config.get("away_hold_sec", 2.0)),
                )
                if self._identity_monitor.initialize():
                    self._identity_storage = IdentityStorage(self.paths)
                    logger.info("身份监控已内嵌到 tracker 进程（复用 tracks）")
                else:
                    self._identity_monitor = None
                    logger.warning("身份监控内嵌初始化失败，回退为几何工位分配")

            # 初始化凝视处理器（modules.gaze=false 时跳过，便于只跑身份/跟踪链路）
            gaze_config = self.config.get("gaze", {})
            self._gaze_processor = None
            if self.config.get("modules", {}).get("gaze", True):
                self._gaze_processor = GazeModule(
                    head_model_path=self.paths.get_model_path(
                        "gaze", gaze_config.get("head_model", _DEFAULT_HEAD_MODEL)
                    ),
                    gaze_model_path=self.paths.get_model_path(
                        "gaze", gaze_config.get("gaze_model", _DEFAULT_GAZE_MODEL)
                    ),
                    roi_json_path=str(
                        self._scene.roi_path
                        if self._scene and self._scene.roi_path
                        else self.paths.base_dir / "data" / "ROI.json"
                    ),
                    config=gaze_config,
                    inference_fn=self.push_display,
                    event_bus=self.event_bus,
                    progress_fn=lambda current, total: (
                        self.inference_stream.update_module_time("gaze", current)
                    ),
                    paths=self.paths,
                )
            else:
                logger.warning("modules.gaze=false，跳过凝视处理器初始化")

            # 初始化举手检测器（实现位于 behavior 模块，tracker 仅持有实例用于前置视角调用）
            cooldown_sec = float(supervision_config.get("cooldown_sec", 1.5))
            fps = float(self.config.get("fps", 30.0))
            self._hand_raiser = HandRaiser(
                detector=self._detector,
                consec_raise=int(supervision_config.get("consec_raise", 2)),
                consec_idle=int(supervision_config.get("consec_idle", 3)),
                cooldown_frames=int(fps * cooldown_sec),
                frame_step=int(supervision_config.get("hand_frame_step", 3)),
            )

            # 初始化结果存储管理器
            self._result_storage = TrackerStorage(self.paths)

            # 订阅流程与规则事件
            self.event_bus.subscribe(
                EventTopic.FLOW_STARTED, self._on_flow_started
            )
            self.event_bus.subscribe(
                EventTopic.FLOW_ENDED, self._on_flow_ended
            )
            self.event_bus.subscribe(
                EventTopic.RULE_KEY_MOMENT, self._on_rule_key_moment
            )

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
            video_path: 视频文件路径，为空时使用配置中的默认前置摄像头视频.
        """
        try:
            if not video_path:
                video_path = str(
                    self.paths.base_dir
                    / self.config.get("videos", {}).get(
                        "front", "data/videos/camFRONT.mpg"
                    )
                )
            elif (
                not Path(video_path).is_absolute()
                and not Path(video_path).exists()
            ):
                video_path = str(self.paths.base_dir / video_path)

            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return

            raw_fps = video_capture.get(cv2.CAP_PROP_FPS)
            if not raw_fps or raw_fps <= 0:
                raise RuntimeError(
                    f"读取 front 视频帧率失败(CAP_PROP_FPS={raw_fps}), 不使用默认帧率"
                )
            fps = float(raw_fps)
            # mpegts 经 -c copy 裁剪会给出 90000 这类畸形帧率, 直接用会让时间轴全乱
            if fps > MAX_REASONABLE_FPS:
                fallback_fps = float(self.config.get("fps", 30.0))
                logger.warning(
                    f"front 视频帧率异常(CAP_PROP_FPS={fps}), "
                    f"回退配置帧率 {fallback_fps}"
                )
                fps = fallback_fps
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            # 确保关键帧目录已准备就绪
            self._result_storage.get_key_frames_dir(self._run_id)

            identities_broadcasted = False

            logger.info(f"开始处理视频: {video_path}")
            logger.info(f"FPS={fps}, 总帧数={total_frames}")

            # 视觉流编码器:把带标注的帧编码为 fMP4(纯视频)推 Redis Stream
            # 前端 MSE 统一按 avc1(无音频)声明, 音频由 voice 模块独立读取原音轨,
            # 若此处带音频会造成 init 段含 mp4a 与前端声明不符 -> 黑屏
            self._vis_encoder = VisEncoder(
                view="front",
                video_path=video_path,
                fps=fps,
                with_audio=False,
                redis_host=self.config.get("redis_host", "localhost"),
                redis_port=self.config.get("redis_port", 6379),
                redis_db=self.config.get("redis_db", 0),
            )

            # 通知前端启动视频流
            self.push_display(
                "video_start", {"localSec": 0, "tag": "start", "data": {}}
            )

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
                self.inference_stream.update_module_snapshot(
                    "tracker",
                    {
                        "roles": dict(self._identity_map),
                        "localSec": round(timestamp, 2),
                    },
                )

                # 1. 目标检测
                high_conf_detections, low_conf_detections = (
                    self._detector.detect_two_thresholds(frame)
                )
                detections = high_conf_detections + low_conf_detections

                # 2. 目标跟踪
                tracks = self._tracker.track(frame, detections)

                # 3. 身份：内嵌模式走人脸锁定（复用 tracks），否则退回几何工位分配
                if self._identity_monitor is not None:
                    locked = self._identity_monitor.process_frame(
                        frame, tracks, timestamp
                    )
                    role_map = {
                        role: tid for role, tid in locked.items() if tid is not None
                    }
                    # 同步回跟踪器：让 track.identity / get_track_by_identity 生效，
                    # 否则 _monitor_distance（LEADER-ROAD 距离）与监护规则拿不到身份
                    self._tracker.identity_map = dict(role_map)
                    self._tracker.identities_assigned = bool(role_map)
                    tid_to_role = {tid: role for role, tid in role_map.items()}
                    for track in tracks:
                        track.identity = tid_to_role.get(int(track.track_id))

                    if role_map != self._identity_map:
                        self._identity_map = role_map
                        roles_formatted = ",".join(
                            f"{role}:{tid}" for role, tid in role_map.items()
                        )
                        if role_map:
                            role_record = {
                                "localSec": round(timestamp, 2),
                                "key_moment": f"角色分配,{roles_formatted}",
                                "source": "identity",
                                "identities": dict(role_map),
                            }
                            self._events.append(dict(role_record))
                            # 同步进 identity 产物，保证 identity_key_moments.json 自洽
                            self._identity_monitor.events.append(
                                dict(role_record)
                            )
                        self.push_display(
                            "identity",
                            {
                                "localSec": round(timestamp, 2),
                                "tag": "ROLE_ASSIGNED",
                                "data": {"roles": role_map},
                            },
                        )
                        logger.info(f"角色锁定更新: {role_map}")

                    for record in self._identity_monitor.drain_events():
                        self._events.append(
                            {
                                "localSec": record["localSec"],
                                "key_moment": record["key_moment"],
                                "source": "identity",
                            }
                        )
                        self.push_display(
                            "identity",
                            {
                                "localSec": record["localSec"],
                                "tag": "IDENTITY_AWAY",
                                "data": {
                                    "identity": record.get("identity"),
                                    "event": record.get("event"),
                                    "duration_sec": record.get("duration_sec"),
                                },
                            },
                        )
                        self.push_event(
                            EventTopic.IDENTITY_AWAY,
                            {
                                "localSec": record["localSec"],
                                "identity": record.get("identity"),
                                "event": record.get("event"),
                                "duration_sec": record.get("duration_sec"),
                                "room_id": self._scene.room_id if self._scene else "",
                            },
                            timestamp=float(record["localSec"]),
                        )
                elif (
                    not self._tracker.identities_assigned
                    and self._tracker.initialized
                    and len(tracks) >= 2
                ):
                    self._tracker.assign_identities(tracks)

                # 几何分配模式下的事件广播与落盘
                if (
                    self._identity_monitor is None
                    and self._tracker.identities_assigned
                    and not identities_broadcasted
                ):
                    identities_broadcasted = True
                    self._identity_map = dict(self._tracker.identity_map)

                    roles_formatted = ",".join(
                        f"{role_name}:{track_id}"
                        for role_name, track_id in self._identity_map.items()
                    )
                    self._events.append(
                        {
                            "localSec": round(timestamp, 2),
                            "key_moment": f"角色分配,{roles_formatted}",
                            "source": "tracker",
                        }
                    )
                    if self._result_storage:
                        self._result_storage.save_key_moments(
                            self._run_id, self._events
                        )
                        self._result_storage.save_role_info(
                            self._run_id, self._identity_map
                        )

                    # 推理流：前端展示身份分配结果
                    self.push_display(
                        "tracking",
                        {
                            "localSec": round(timestamp, 2),
                            "tag": "ROLE_ASSIGNED",
                            "data": {"roles": self._identity_map},
                        },
                    )

                    # 保存身份分配关键帧
                    self._result_storage.save_key_frame(
                        self._run_id,
                        f"role_assigned_{timestamp:.1f}s.jpg",
                        frame,
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
                        self._run_id,
                        f"hand_raise_{label}_{timestamp:.1f}s.jpg",
                        frame,
                    )

                    tag_label = (
                        f"身份={identity}" if identity else "未分配身份"
                    )
                    logger.info(
                        f"举手检测: {tag_label} "
                        f"(track_id={track_id}) @{timestamp:.1f}s"
                    )

                # 5. 距离与人数监控（每 30 帧检查一次）
                if frame_count % 30 == 0:
                    self._monitor_distance(timestamp)
                    self._monitor_headcount(timestamp, tracks)

                # 6. 标注叠加到帧 + 喂入视觉流编码器(标注画进帧,fMP4 推送,前端 MSE 直接播)
                try:
                    # 凝视: process_frame 用原始 frame 推理 + 绘制 ROI/head/gaze
                    # 到 frame 副本并返回
                    if self._gaze_processor:
                        annotated_frame = self._gaze_processor.process_frame(
                            frame, timestamp, frame_count
                        )
                    else:
                        annotated_frame = frame.copy()
                    # 跟踪框 + 身份标签 + 离岗叠加层
                    annotated_frame = draw_tracks(
                        annotated_frame,
                        tracks,
                        self._identity_map,
                        (
                            self._identity_monitor.snapshot
                            if self._identity_monitor is not None
                            else None
                        ),
                    )
                    # LEADER 与 ROAD 距离线
                    self._draw_distance_lines(annotated_frame)
                    # 举手标签(在对应跟踪框上方)
                    for track_id, identity in raised_targets:
                        track = next(
                            (
                                candidate
                                for candidate in tracks
                                if candidate.track_id == track_id
                            ),
                            None,
                        )
                        if track is not None:
                            bbox = track.bbox
                            cv2.putText(
                                annotated_frame,
                                "举手",
                                (int(bbox[0]), max(20, int(bbox[1]) - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                            )
                    # 喂入编码器(惰性启动,首帧确定分辨率)
                    if self._vis_encoder is not None:
                        self._vis_encoder.feed_frame(annotated_frame)
                except Exception as vis_error:
                    logger.warning(f"视觉帧叠加失败: {vis_error}")

                # 进度日志
                if frame_count % 300 == 0:
                    percentage = (
                        frame_count * 100 // total_frames
                        if total_frames
                        else 0
                    )
                    logger.info(
                        f"{frame_count}/{total_frames}帧 {percentage}%"
                    )

            # 收尾：闭合未结束的离岗段
            if self._identity_monitor is not None:
                self._identity_monitor.close(timestamp)
                for record in self._identity_monitor.drain_events():
                    self._events.append(
                        {
                            "localSec": record["localSec"],
                            "key_moment": record["key_moment"],
                            "source": "identity",
                        }
                    )
                    # 收尾补发的离岗段也要进事件总线，否则规则层漏判最后一段
                    self.push_event(
                        EventTopic.IDENTITY_AWAY,
                        {
                            "localSec": record["localSec"],
                            "identity": record.get("identity"),
                            "event": record.get("event"),
                            "duration_sec": record.get("duration_sec"),
                            "room_id": (
                                self._scene.room_id if self._scene else ""
                            ),
                        },
                        timestamp=float(record["localSec"]),
                    )

            video_capture.release()
            if self._vis_encoder is not None:
                self._vis_encoder.finalize()
            logger.info(f"视频处理完成，共 {frame_count} 帧")

        except Exception as process_error:
            logger.error(f"视频处理失败: {process_error}", exc_info=True)

    def save_results(self, run_id: str) -> None:
        """保存 MOT 结果（委托给 TrackerStorage + gaze）.

        Args:
            run_id: 本次运行标识.
        """
        self._result_storage.save_key_moments(run_id, self._events)
        self._result_storage.save_role_info(run_id, self._identity_map)
        if self._gaze_processor is not None:
            self._gaze_processor.save_results(run_id)
        if self._identity_monitor is not None and self._identity_storage is not None:
            self._identity_storage.save_key_moments(
                run_id, self._identity_monitor.events
            )
            self._identity_storage.save_away_events(
                run_id, self._identity_monitor.away_events
            )
            self._identity_storage.save_role_info(
                run_id,
                self._identity_monitor.snapshot,
                self._scene.room_id if self._scene else "",
            )
        logger.info(f"MOT 结果已保存到 {run_id}")

    def _on_flow_started(self, event: dict) -> None:
        """订阅 FLOW_STARTED：监护制流程开始时触发，载荷位于 "data" 键.

        Args:
            event: 事件总线分发的标准事件字典.
        """
        payload = event.get("data", {})
        if payload.get("flow_type") == "supervision":
            self._supervision_active = True
            flow_start_timestamp = payload.get("flow_start_sec", 0)
            logger.info(
                f"Tracker: 监护制流程开始 @{flow_start_timestamp:.1f}s"
            )

    def _on_flow_ended(self, event: dict) -> None:
        """订阅 FLOW_ENDED：流程结束时触发，载荷位于 "data" 键.

        Args:
            event: 事件总线分发的标准事件字典.
        """
        payload = event.get("data", {})
        if payload.get("flow_type") == "supervision":
            self._supervision_active = False
            flow_end_timestamp = payload.get("flow_end_sec", 0)
            logger.info(f"Tracker: 监护制流程结束 @{flow_end_timestamp:.1f}s")

    def _on_rule_key_moment(self, event: dict) -> None:
        """订阅 RULE_KEY_MOMENT：规则层下发的监护绑定/解绑 key_moment 归属 tracker.

        时间戳优先取载荷 "localSec"，缺省回退到事件信封 "ts".

        Args:
            event: 事件总线分发的标准事件字典.
        """
        payload = event.get("data", {})
        timestamp = payload.get("localSec", event.get("ts", 0))
        key_moment = payload.get("key_moment", "")
        if not key_moment:
            return
        self._events.append(
            {
                "localSec": round(timestamp, 2),
                "key_moment": key_moment,
                "source": "tracker",
            }
        )
        if self._result_storage:
            self._result_storage.save_key_moments(self._run_id, self._events)
        logger.debug(
            f"Tracker 接收 RULE_KEY_MOMENT: {key_moment} @{timestamp:.1f}s"
        )

    def update_progress(
        self, current: float, total: Optional[float] = None
    ) -> None:
        """更新 tracker 进度并推送 gaze 前端进度条.

        Args:
            current: 当前处理到的时间点（秒）.
            total: 视频总时长（秒）.
        """
        super().update_progress(current, total)
        if total and total > 0:
            now = time.time()
            if not hasattr(self, "_last_gaze_progress_push"):
                self._last_gaze_progress_push = 0
            if now - self._last_gaze_progress_push >= 1.0:
                self._last_gaze_progress_push = now
                gaze_percentage = min(100.0, current / total * 100)
                self.push_display(
                    "progress",
                    {
                        "localSec": round(current, 2),
                        "tag": "progress",
                        "data": {
                            "label": "gaze",
                            "pct": round(gaze_percentage, 1),
                        },
                    },
                )

    def _monitor_distance(self, timestamp: float) -> None:
        """距离监控：判定 LEADER 与 ROAD1/ROAD2 距离状态并推送.

        Args:
            timestamp: 当前帧时间点（秒）.
        """
        if not self._tracker.identities_assigned:
            return
        leader = self._tracker.get_track_by_identity("LEADER")
        if not leader:
            return
        for road_name in ("ROAD1", "ROAD2"):
            target = self._tracker.get_track_by_identity(road_name)
            if not target:
                continue
            distance = float(
                np.linalg.norm(leader.get_center() - target.get_center())
            )
            dist_close = self.config.get("supervision", {}).get(
                "dist_close_px", 200
            )
            dist_near = self.config.get("supervision", {}).get(
                "dist_near_px", 560
            )

            if distance <= dist_close:
                state_label = "监护中"
            elif distance <= dist_near:
                state_label = "接近中"
            else:
                state_label = "未监护"

            last_state = self._last_supervision_states.get(road_name)
            is_state_changed = state_label != last_state
            self._last_supervision_states[road_name] = state_label

            event_data = {
                "localSec": round(timestamp, 2),
                "tag": "SUPERVISOR_STATUS",
                "data": {
                    "state": state_label,
                    "operator": road_name,
                    "distance_px": int(distance),
                },
            }

            if self._supervision_active:
                self.push_display("tracking", event_data)

            if self._supervision_active or is_state_changed:
                self.push_event(
                    EventTopic.TRACKER_PROXIMITY,
                    {
                        "localSec": round(timestamp, 2),
                        "state": state_label,
                        "operator": road_name,
                        "distance_px": int(distance),
                    },
                    timestamp=timestamp,
                )

    def _monitor_headcount(
        self, timestamp: float, tracks: List[STrack]
    ) -> None:
        """人数监控：人数变化时推送，无人值守时保存 key_moment.

        Args:
            timestamp: 当前帧时间点（秒）.
            tracks: 当前帧中的跟踪目标列表.
        """
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
            people_data["data"][
                "state_alert"
            ] = "提醒：当前主控室只有一人，请注意安全！"
            self.push_display("tracking", people_data)
        elif people_count < 1:
            people_data["data"]["state"] = "主控室无人值守"
            people_data["data"]["state_alert"] = "警告：当前主控室无人值守！"
            self.push_display("tracking", people_data)
        else:
            self.push_display("tracking", people_data)

        if people_count < 1:
            self._events.append(
                {
                    "localSec": round(timestamp, 2),
                    "key_moment": "主监控室少于1人",
                    "source": "tracker",
                }
            )
            if self._result_storage:
                self._result_storage.save_key_moments(
                    self._run_id, self._events
                )
            self.push_event(
                EventTopic.TRACKER_HEADCOUNT,
                {
                    "localSec": round(timestamp, 2),
                    "count": people_count,
                },
                timestamp=timestamp,
            )

    def _draw_distance_lines(self, annotated_frame: np.ndarray) -> None:
        """在可视化帧上绘制 LEADER 与 ROAD1/ROAD2 的距离线.

        Args:
            annotated_frame: 待就地绘制距离线的可视化帧.
        """
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
                annotated_frame,
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
                annotated_frame,
                f"{road_name} {int(distance)}px",
                mid_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )
