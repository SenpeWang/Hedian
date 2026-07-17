"""
多目标跟踪模块入口

继承 BaseModule，实现统一接口。
"""
import os
import json
import logging
import numpy as np

from core.base_module import BaseModule
from core.event_bus import EventBus, EventTopic
from core.display_buffer import DisplayBuffer
from core.path_manager import PathConfig

from modules.tracker.object_detector import ObjectDetector
from modules.tracker.multi_object_tracker import MultiObjectTracker
from modules.tracker.hand_raiser import HandRaiser
from modules.tracker.storage import TrackerStorage
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
        display_buffer: DisplayBuffer,
    ):
        super().__init__(event_bus, config, paths, display_buffer)
        self._detector = None
        self._tracker = None
        self._hand_raiser = None
        # SM 已移到 rules
        self._result_storage = None
        self._events = []
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

            # 初始化凝视处理器（独立模块，MOT调用）
            gaze_config = self.config.get("gaze", {})
            self._gaze_processor = GazeModule(
                head_model_path=self.paths.get_model_path("gaze", "yolov8n_head.onnx"),
                gaze_model_path=self.paths.get_model_path("gaze", "gazelle_dinov3_vits16plus_finetuned_1x3x640x640_1xNx4.onnx"),
                roi_json_path=str(self.paths.base_dir / "data" / "ROI.json"),
                config=gaze_config,
                display_fn=self.push_display,
                event_bus=self.event_bus,
                progress_fn=lambda cur, total: self.display_buffer.update_module_time("gaze", cur),
            )
            

            # 盘台监控状态
            self._panel_away_start_ts = -1.0
            self._panel_away_start_frame = -1
            self._panel_violation_records = []
            self._panel_both_away = False
            self._PANEL_AWAY_THRESHOLD = 60.0
            self._PANEL_MATCH_DIST = 150.0


            # 初始化举手检测器
            self._hand_raiser = HandRaiser(
                detector=self._detector,
                vote_window=supervision_config.get("vote_window", 3),
                vote_threshold=supervision_config.get("vote_threshold", 2),
                cooldown_frames=supervision_config.get("cooldown_frames", 300),
            )

            # 初始化结果存储
            self._result_storage = TrackerStorage(self.paths)

            # Redis 连接（用于检查监护制状态）
            import redis
            self._redis = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

            logger.info("MOT 模块初始化完成")
            # 监护制流程状态（通过订阅 event_bus 事件更新）
            return True

        except Exception as e:
            logger.error(f"MOT 模块初始化失败: {e}", exc_info=True)
            return False

    def process_video(self, video_path: str) -> None:
        """处理视频，进行目标检测和跟踪"""
        import cv2
        import queue as queue_mod

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

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or self.config.get("fps", 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            # 创建关键帧目录（保存到 run_id 目录下）
            kf_dir = self.paths.get_result_dir(self._run_id, "tracker") / "key_frames"
            kf_dir.mkdir(parents=True, exist_ok=True)

            role_assigned_done = False

            logger.info(f"开始处理视频: {video_path}")
            logger.info(f"FPS={fps}, 总帧数={total_frames}")

            # 通知前端启动视频流
            self.push_display("video_start", {})

            tracks = []  # 初始化，确保可视化代码始终可用

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                ts = frame_count / fps
                frame_count += 1

                # 更新进度
                self.update_progress(ts, total_frames / fps)
                
                # 同步推送 gaze 模块的推理进度给前端右上角展示
                gaze_pct = min(ts / (total_frames / fps) * 100, 100) if total_frames > 0 else 0
                self.push_display("progress", {
                    "localSec": round(ts, 2),
                    "label": "gaze",
                    "pct": round(gaze_pct, 1),
                })

                # 更新聚合器快照（供前端 context 使用）
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
                        "source": "tracker",
                    })

                    # 推理流：前端展示角色分配结果
                    self.push_display("tracking", {
                        "localSec": round(ts, 2),
                        "event": "ROLE_ASSIGNED",
                        "roles_details": self._roles_info,
                    })
                    # 事件流：通知规则状态机角色已分配
                    self.push_event(EventTopic.TRACKER_ROLE_ASSIGNED, {
                        "localSec": round(ts, 2),
                        "roles": self._roles_info,
                    }, ts=ts)

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
                    # 推理流：前端展示举手检测
                    self.push_display("tracking", {
                        "localSec": round(ts, 2),
                        "event": "HAND_RAISED",
                        "state": "举手",
                        "operator": hand_role,
                        "roles_details": self._roles_info,
                    })
                    # 事件流：通知规则状态机有人举手
                    self.push_event(EventTopic.TRACKER_HAND_RAISED, {
                        "localSec": round(ts, 2),
                        "operator": hand_role,
                    }, ts=ts)

                    # 记录监护请求事件
                    self._events.append({
                        "localSec": round(ts, 2),
                        "key_moment": "请求监护",
                        "ID": hand_role,
                        "source": "tracker",
                    })

                    # 保存关键帧
                    kf_path = kf_dir / f"hand_raise_{hand_role}_{ts:.1f}s.jpg"
                    cv2.imwrite(str(kf_path), frame)

                    self._events.append({
                        "localSec": round(ts, 2),
                        "key_moment": "举手",
                        "ID": hand_role,
                        "source": "tracker",
                    })

                    logger.info(f"举手检测: {hand_role} @{ts:.1f}s")

                # 距离与人数监控（每30帧检查一次）
                if frame_count % 30 == 0:
                    # 1. 距离监控（需要已分配角色）
                    if self._tracker.roles_assigned:
                        leader = self._tracker.get_track_by_role("LEADER")
                        # 检查 ROAD1 和 ROAD2 的距离
                        for road_name in ("ROAD1", "ROAD2"):
                            target = self._tracker.get_track_by_role(road_name)
                            if leader and target:
                                d = float(np.linalg.norm(leader.get_center() - target.get_center()))
                                dist_close = self.config.get("supervision", {}).get("dist_close_px", 280)
                                dist_near = self.config.get("supervision", {}).get("dist_near_px", 560)

                                if d <= dist_close:
                                    state_label = "监护中"
                                elif d <= dist_near:
                                    state_label = "接近中"
                                else:
                                    state_label = "未监护"

                                # 判定状态是否发生改变（作为消息流的关键事件）
                                last_state = self._last_supervision_states.get(road_name, None)
                                is_state_changed = (state_label != last_state)
                                self._last_supervision_states[road_name] = state_label

                                event_data = {
                                    "localSec": round(ts, 2),
                                    "event": "SUPERVISOR_STATUS",
                                    "state": state_label,
                                    "operator": road_name,
                                    "distance_px": int(d),
                                    "roles_details": self._roles_info,
                                }

                                # 1. 普通推理流推送（只有在监护制激活时才推送到前端展示）
                                supervision_active = self._redis.get("supervision:active")
                                if supervision_active == "true":
                                    self.push_display("tracking", event_data)

                                # 2. 消息流推送（只在监护状态改变时发布，用于跨模块的制度逻辑判定）
                                if is_state_changed:
                                    self.push_event(EventTopic.TRACKER_PROXIMITY, {
                                        "localSec": round(ts, 2),
                                        "state": state_label,
                                        "operator": road_name,
                                        "distance_px": int(d),
                                    }, ts=ts)

                                # 保存事件到本地记录
                                self._events.append({
                                    "localSec": round(ts, 2),
                                    "key_moment": state_label,
                                    "ID": road_name,
                                    "source": "tracker",
                                })

                    # 2. 人数监控（主控室人员状态）
                    people_count = len(tracks)
                    last_count = getattr(self, "_last_people_count", -1)
                    if people_count != last_count:
                        self._last_people_count = people_count
                        
                        people_data = {
                            "localSec": round(ts, 2),
                            "event": "PEOPLE_COUNT_UPDATE",
                            "count": people_count,
                        }
                        
                        # 特殊提醒和警报（写入推理流，触发前端展示）
                        if people_count == 1:
                            people_data["state"] = "主控室仅有1人"
                            people_data["state_alert"] = "提醒：当前主控室只有一人，请注意安全！"
                            self.push_display("tracking", people_data)
                        elif people_count < 1:
                            people_data["state"] = "主控室无人值守"
                            people_data["state_alert"] = "警告：当前主控室无人值守！"
                            self.push_display("tracking", people_data)
                        else:
                            # 正常情况（>=2人），仅更新人数数据
                            self.push_display("tracking", people_data)

                        # 发送给跨进程消息总线（消息流），供制度层进行违规记录
                        self.push_event(EventTopic.TRACKER_HEADCOUNT, {
                            "localSec": round(ts, 2),
                            "count": people_count,
                        }, ts=ts)


                # 可视化：在帧上画跟踪结果 + 距离 + 注视结果
                if frame_count % 2 == 0:
                    try:
                        vis_frame = frame.copy()
                        # 1. 画跟踪框和角色
                        vis_frame = draw_tracks(vis_frame, tracks, self._roles_info)
                        # 2. 画距离线
                        if self._tracker.roles_assigned:
                            leader = self._tracker.get_track_by_role("LEADER")
                            if leader:
                                # 画灰色线到 ROAD1/ROAD2
                                for _rn in ("ROAD1", "ROAD2"):
                                    _rt = self._tracker.get_track_by_role(_rn)
                                    if _rt:
                                        lc = leader.get_center()
                                        rc = _rt.get_center()
                                        d = float(np.linalg.norm(lc - rc))
                                        cv2.line(vis_frame, tuple(map(int, lc)), tuple(map(int, rc)), (128, 128, 128), 2)
                                        mid = ((int(lc[0]) + int(rc[0])) // 2, (int(lc[1]) + int(rc[1])) // 2)
                                        cv2.putText(vis_frame, f"{_rn} {int(d)}px", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                        # 3. 画注视结果（ROI + 头部框 + 注视线 + 告警）
                        vis_frame = self._draw_gaze(vis_frame, frame_count, ts, tracks)
                        _, jpeg = cv2.imencode(".jpg", vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

                        # 通过 DisplayBuffer 推送视频帧（与其他事件对齐）
                        import base64
                        self.push_display("video_frame", {
                            "localSec": round(ts, 2),
                            "frame_data": base64.b64encode(jpeg.tobytes()).decode('utf-8'),
                            "frame_id": frame_count,
                            "source": "tracker",
                        })

                        # 也推送到 frame_queue（供实时视频流使用）
                        if frame_queue is not None:
                            try:
                                frame_queue.put_nowait(jpeg.tobytes())
                            except queue_mod.Full:
                                try:
                                    frame_queue.get_nowait()
                                except queue_mod.Empty:
                                    pass
                                frame_queue.put_nowait(jpeg.tobytes())
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
        """保存 MOT 结果"""
        if not self._events:
            logger.warning("没有事件可保存")
            return

        # 保存关键事件
        self._result_storage.save_key_moments(run_id, self._events)

        # 保存角色信息
        self._result_storage.save_role_info(run_id, self._roles_info)

        # 保存盘台违规记录
        if self._panel_violation_records:
            self._result_storage.save_panel_violations(run_id, self._panel_violation_records)

        # 保存注视告警事件 (gaze_key_moments.json)
        try:
            gaze_events = self._gaze_processor.get_events()
            if gaze_events:
                gaze_dir = self.paths.result_root / run_id / "gaze"
                gaze_dir.mkdir(parents=True, exist_ok=True)
                gaze_path = gaze_dir / "gaze_key_moments.json"
                with open(gaze_path, "w", encoding="utf-8") as f:
                    json.dump(gaze_events, f, ensure_ascii=False, indent=2)
                logger.info(f"保存 {len(gaze_events)} 个注视告警事件到 {gaze_path}")
            else:
                logger.info("没有注视告警事件，跳过保存")
        except Exception as e:
            logger.error(f"保存注视告警事件失败: {e}", exc_info=True)

        logger.info(f"MOT 结果已保存到 {run_id}")

    def update_progress(self, current: float, total: float = None) -> None:
        """更新人员追踪和注视检测的共同进度，防止奇数帧导致的进度差异悬挂"""
        super().update_progress(current, total)
        self.display_buffer.update_module_time("gaze", current)

    def _draw_gaze(self, frame, frame_count, ts, tracks=None):
        """调用凝视处理器：检测、估计、可视化、推送"""
        return self._gaze_processor.process_frame(frame, ts, frame_count)
