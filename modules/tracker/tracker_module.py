"""
多目标跟踪模块入口

继承 BaseModule，实现统一接口。
"""
import os
import logging
import time
import base64
import numpy as np

from core.base_module import BaseModule
from core.event_bus import EventBus, EventTopic
from core.inference_bus import InferenceBus
from core.path_manager import PathConfig

from modules.tracker.object_detector import ObjectDetector
from modules.tracker.multi_object_tracker import MultiObjectTracker
from modules.behavior.hand_raiser import HandRaiser
from modules.tracker.storage_tracker import TrackerStorage
from modules.tracker.visualizer import draw_tracks, draw_supervision
from modules.gaze.gaze_module import GazeModule

logger = logging.getLogger("module.tracker")


class TrackerModule(BaseModule):
    """
    多目标跟踪模块

    负责目标检测、跟踪、举手检测和监护状态管理。
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
        self._tracker = None
        self._hand_raiser = None
        self._result_storage = None
        self._events = []                # tracker 自身的关键时刻事件
        self._roles_info = {}
        self._last_supervision_states = {}

    @property
    def module_name(self) -> str:
        return "tracker"

    def initialize(self) -> bool:
        """初始化 MOT 模块"""
        try:
            # 获取配置
            tracker_config = self.config.get("tracker", {})
            supervision_config = self.config.get("supervision", {})

            # 初始化检测器
            detection_config = tracker_config.get("detection", {})
            self._detector = ObjectDetector(
                model_path=self.paths.get_model_path("detection", "yolo11_MOT.pt"),
                pose_model_path=self.paths.get_model_path("detection", "yolo26s-pose.pt"),
                conf_threshold=detection_config.get("conf_threshold", 0.65),
                pose_confidence=detection_config.get("pose_confidence", 0.3),
                nms_threshold=detection_config.get("nms_threshold", 0.35),
                img_size=detection_config.get("img_size", 640),
            )

            # 初始化跟踪器
            self._tracker = MultiObjectTracker()

            # 初始化凝视处理器
            gaze_config = self.config.get("gaze", {})
            self._gaze_processor = GazeModule(
                head_model_path=self.paths.get_model_path("gaze", "yolov8n_head.onnx"),
                gaze_model_path=self.paths.get_model_path("gaze", "gazelle_dinov3_vits16plus_finetuned_1x3x640x640_1xNx4.onnx"),
                roi_json_path=str(self.paths.base_dir / "data" / "ROI.json"),
                config=gaze_config,
                display_fn=self.push_display,
                event_bus=self.event_bus,
                progress_fn=lambda cur, total: self.display_buffer.update_module_time("gaze", cur),
                paths=self.paths,
            )


            # 初始化举手检测器
            self._hand_raiser = HandRaiser(
                detector=self._detector,
                vote_window=supervision_config.get("vote_window", 3),
                vote_threshold=supervision_config.get("vote_threshold", 2),
                cooldown_frames=supervision_config.get("cooldown_frames", 300),
            )

            # 初始化结果存储
            self._result_storage = TrackerStorage(self.paths)

            # 监护制流程状态
            self._supervision_active = False

            self.event_bus.subscribe(EventTopic.FLOW_STARTED, self._on_flow_started)
            self.event_bus.subscribe(EventTopic.FLOW_ENDED, self._on_flow_ended)
            # 规则层判断的监护绑定/解绑 key_moment 归属 tracker
            self.event_bus.subscribe(EventTopic.RULE_KEY_MOMENT, self._on_rule_key_moment)

            logger.info("MOT 模块初始化完成")
            return True

        except Exception as e:
            logger.error(f"MOT 模块初始化失败: {e}", exc_info=True)
            return False

    def process_video(self, video_path: str) -> None:
        """处理视频，进行目标检测和跟踪"""
        import cv2

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or self.config.get("fps", 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            # 创建关键帧目录
            kf_dir = self.paths.get_result_dir(self._run_id, "tracker") / "key_frames"
            kf_dir.mkdir(parents=True, exist_ok=True)

            role_assigned_done = False

            logger.info(f"开始处理视频: {video_path}")
            logger.info(f"FPS={fps}, 总帧数={total_frames}")

            # 通知前端启动视频流
            self.push_display("video_start", {"localSec": 0, "tag": "start", "data": {}})

            tracks = []  # 初始化，确保可视化代码始终可用

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                ts = frame_count / fps
                frame_count += 1

                # 更新进度
                self.update_progress(ts, total_frames / fps)

                # 更新聚合器快照
                self.display_buffer.update_module_snapshot("tracker", {
                    "roles": dict(self._roles_info),
                    "localSec": round(ts, 2),
                })

                # 目标检测
                high_dets, low_dets = self._detector.detect_two_thresholds(frame)
                detections = high_dets + low_dets
                

                # 目标跟踪
                tracks = self._tracker.track(frame, detections)

                # 角色分配
                if not self._tracker.roles_assigned and self._tracker.initialized and len(tracks) >= 2:
                    self._tracker.assign_roles(tracks)

                # 角色分配事件
                if self._tracker.roles_assigned and not role_assigned_done:
                    role_assigned_done = True
                    self._roles_info = dict(self._tracker.role_map)

                    roles_str = ",".join(f"{k}:{v}" for k, v in self._roles_info.items())
                    self._events.append({
                        "localSec": round(ts, 2),
                        "key_moment": f"角色分配,{roles_str}",
                    })

                    # 推理流：前端展示角色分配结果
                    self.push_display("tracking", {
                        "localSec": round(ts, 2),
                        "tag": "ROLE_ASSIGNED",
                        "data": {"roles": self._roles_info}
                    })

                    # 保存关键帧
                    kf_path = kf_dir / f"role_assigned_{ts:.1f}s.jpg"
                    cv2.imwrite(str(kf_path), frame)

                    logger.info(f"角色分配: {self._roles_info}")

                # 举手检测
                roles = {}
                if self._tracker.roles_assigned:
                    for role_name in ("LEADER", "ROAD1", "ROAD2"):
                        role_track = self._tracker.get_track_by_role(role_name)
                        if role_track:
                            roles[role_name] = role_track

                hand_role = self._hand_raiser.check(
                    frame, tracks, self._tracker.roles_assigned, frame_count, roles
                )

                if hand_role:
                    # 举手事件归属 behavior
                    self.push_display("behavior", {
                        "localSec": round(ts, 2),
                        "tag": "HAND_RAISED",
                        "data": {"state": "举手", "operator": hand_role}
                    })
                    self.push_event(EventTopic.BEHAVIOR_HAND_RAISED, {
                        "localSec": round(ts, 2),
                        "operator": hand_role,
                    }, ts=ts)

                    # 保存关键帧
                    kf_path = kf_dir / f"hand_raise_{hand_role}_{ts:.1f}s.jpg"
                    cv2.imwrite(str(kf_path), frame)

                    logger.info(f"举手检测: {hand_role} @{ts:.1f}s")

                # 距离与人数监控（每30帧检查一次）
                if frame_count % 30 == 0:
                    self._monitor_distance(ts)
                    self._monitor_headcount(ts, tracks)


                # 可视化：在帧上画跟踪结果 + 距离 + 注视结果（每2帧一次）
                if frame_count % 2 == 0:
                    try:
                        # 直接在原 frame 上绘制（下游不再使用 frame）
                        vis_frame = draw_tracks(frame, tracks, self._roles_info)
                        if self._tracker.roles_assigned:
                            self._draw_distance_lines(vis_frame)
                        vis_frame = self._draw_gaze(vis_frame, frame_count, ts, tracks)
                        # 降分辨率到 720p + JPEG quality 35：大幅降低编码与传输开销
                        vis_small = cv2.resize(vis_frame, (1280, 720))
                        _, jpeg = cv2.imencode(".jpg", vis_small, [cv2.IMWRITE_JPEG_QUALITY, 35])

                        # 推理流推送视频帧（globalSec=ts，前端按 batch 同步对齐）
                        self.push_display("video", {
                            "localSec": round(ts, 2),
                            "tag": "frame",
                            "data": {
                                "frame_data": base64.b64encode(jpeg.tobytes()).decode('utf-8'),
                                "frame_id": frame_count,
                            },
                        })
                    except Exception as e:
                        logger.warning(f"可视化帧失败: {e}")

                # 进度日志
                if frame_count % 300 == 0:
                    pct = frame_count * 100 // total_frames if total_frames else 0
                    logger.info(f"{frame_count}/{total_frames}帧 {pct}%")

            cap.release()
            logger.info(f"视频处理完成，共 {frame_count} 帧")

        except Exception as e:
            logger.error(f"视频处理失败: {e}", exc_info=True)

    def save_results(self, run_id: str) -> None:
        """保存 MOT 结果（委托给 TrackerStorage + gaze）"""
        self._result_storage.save_key_moments(run_id, self._events)
        self._result_storage.save_role_info(run_id, self._roles_info)
        # 举手事件由 behavior 进程统一保存；注视告警事件由 gaze 模块保存
        self._gaze_processor.save_results(run_id)
        logger.info(f"MOT 结果已保存到 {run_id}")

    def _on_flow_started(self, msg: dict) -> None:
        """订阅 FLOW_STARTED：监护制流程开始时触发"""
        data = msg.get("data", {})
        flow_type = data.get("flow_type", "")
        if flow_type == "supervision":
            self._supervision_active = True
            ts = data.get("flow_start_sec", 0)
            logger.info(f"Tracker: 监护制流程开始 @{ts:.1f}s")

    def _on_flow_ended(self, msg: dict) -> None:
        """订阅 FLOW_ENDED：流程结束时触发"""
        data = msg.get("data", {})
        flow_type = data.get("flow_type", "")
        if flow_type == "supervision":
            self._supervision_active = False
            ts = data.get("flow_end_sec", 0)
            logger.info(f"Tracker: 监护制流程结束 @{ts:.1f}s")

    def _on_rule_key_moment(self, msg: dict) -> None:
        """订阅 RULE_KEY_MOMENT：规则层下发的监护绑定/解绑 key_moment 归属 tracker"""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", 0))
        key_moment = data.get("key_moment", "")
        if not key_moment:
            return
        # 强制 source 为 tracker：只有四个基础模块才产生 key_moment
        self._events.append({
            "localSec": round(ts, 2),
            "key_moment": key_moment,
            "source": "tracker",
        })
        logger.debug(f"Tracker 接收 RULE_KEY_MOMENT: {key_moment} @{ts:.1f}s")

    def update_progress(self, current: float, total: float = None) -> None:
        """更新 tracker + gaze 双模块进度"""
        super().update_progress(current, total)
        self.display_buffer.update_module_time("gaze", current)
        if total and total > 0:
            now = time.time()
            if not hasattr(self, "_last_gaze_progress_push"):
                self._last_gaze_progress_push = 0
            if now - self._last_gaze_progress_push >= 1.0:
                self._last_gaze_progress_push = now
                gaze_pct = min(100.0, current / total * 100)
                self.push_display("progress", {
                    "localSec": round(current, 2),
                    "tag": "progress",
                    "data": {"label": "gaze", "pct": round(gaze_pct, 1)},
                })

    def _monitor_distance(self, ts: float) -> None:
        """距离监控：判定 LEADER 与 ROAD1/ROAD2 距离状态并推送"""
        if not self._tracker.roles_assigned:
            return
        leader = self._tracker.get_track_by_role("LEADER")
        if not leader:
            return
        for road_name in ("ROAD1", "ROAD2"):
            target = self._tracker.get_track_by_role(road_name)
            if not target:
                continue
            d = float(np.linalg.norm(leader.get_center() - target.get_center()))
            dist_close = self.config.get("supervision", {}).get("dist_close_px", 200)
            dist_near = self.config.get("supervision", {}).get("dist_near_px", 560)

            if d <= dist_close:
                state_label = "监护中"
            elif d <= dist_near:
                state_label = "接近中"
            else:
                state_label = "未监护"

            last_state = self._last_supervision_states.get(road_name)
            is_state_changed = (state_label != last_state)
            self._last_supervision_states[road_name] = state_label

            event_data = {
                "localSec": round(ts, 2),
                "tag": "SUPERVISOR_STATUS",
                "data": {"state": state_label, "operator": road_name, "distance_px": int(d)},
            }

            # 推理流推送：仅在监护制激活时推送到前端
            if self._supervision_active:
                self.push_display("tracking", event_data)

            # 事件流推送：激活时每秒推送，未激活时仅在状态改变时推送
            if self._supervision_active or is_state_changed:
                self.push_event(EventTopic.TRACKER_PROXIMITY, {
                    "localSec": round(ts, 2),
                    "state": state_label,
                    "operator": road_name,
                    "distance_px": int(d),
                }, ts=ts)

    def _monitor_headcount(self, ts: float, tracks: list) -> None:
        """人数监控：人数变化时推送，无人值守时保存 key_moment"""
        people_count = len(tracks)
        last_count = getattr(self, "_last_people_count", -1)
        if people_count == last_count:
            return
        self._last_people_count = people_count

        people_data = {
            "localSec": round(ts, 2),
            "tag": "PEOPLE_COUNT_UPDATE",
            "data": {"count": people_count},
        }

        # 特殊提醒和警报
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

        # 仅当无人值守时保存为 key_moment 并推送事件流
        if people_count < 1:
            self._events.append({
                "localSec": round(ts, 2),
                "key_moment": "主监控室少于1人",
            })
            self.push_event(EventTopic.TRACKER_HEADCOUNT, {
                "localSec": round(ts, 2),
                "count": people_count,
            }, ts=ts)

    def _draw_distance_lines(self, vis_frame) -> None:
        """在可视化帧上绘制 LEADER 与 ROAD1/ROAD2 的距离线"""
        import cv2
        leader = self._tracker.get_track_by_role("LEADER")
        if not leader:
            return
        for road_name in ("ROAD1", "ROAD2"):
            target = self._tracker.get_track_by_role(road_name)
            if not target:
                continue
            lc = leader.get_center()
            rc = target.get_center()
            d = float(np.linalg.norm(lc - rc))
            cv2.line(vis_frame, tuple(map(int, lc)), tuple(map(int, rc)), (128, 128, 128), 2)
            mid = ((int(lc[0]) + int(rc[0])) // 2, (int(lc[1]) + int(rc[1])) // 2)
            cv2.putText(vis_frame, f"{road_name} {int(d)}px", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    def _draw_gaze(self, frame, frame_count, ts, tracks=None):
        """调用凝视处理器：检测、估计、可视化、推送"""
        return self._gaze_processor.process_frame(frame, ts, frame_count)
