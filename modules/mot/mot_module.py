"""
多目标跟踪模块入口

继承 BaseModule，实现统一接口。
"""
import os
import logging
import numpy as np

from core.base_module import BaseModule
from core.message_bus import MessageBus, MsgType
from core.frontend_sync import FrontendSync
from core.path_manager import PathConfig

from modules.mot.object_detector import ObjectDetector
from modules.mot.multi_object_tracker import MultiObjectTracker
from modules.mot.hand_raiser import HandRaiser
from modules.mot.result_storage import MOTResultStorage
from modules.mot.visualizer import draw_tracks, draw_supervision
from modules.gaze.head_detector import HeadDetector
from modules.gaze.gaze_estimator import GazeEstimator
from modules.gaze.roi_classifier import ROIClassifier

logger = logging.getLogger("module.mot")


class MOTModule(BaseModule):
    """
    多目标跟踪模块

    负责目标检测、跟踪、举手检测和监护状态管理。
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
        self._tracker = None
        self._hand_raiser = None
        # SM 已移到 regulations
        self._result_storage = None
        self._events = []
        self._roles_info = {}

    @property
    def module_name(self) -> str:
        return "mot"

    def initialize(self) -> bool:
        """初始化 MOT 模块"""
        try:
            # 获取配置
            mot_config = self.config.get("mot", {})
            supervision_config = self.config.get("supervision", {})

            # 初始化检测器
            detection_config = mot_config.get("detection", {})
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

            # 初始化注视检测器
            gaze_config = self.config.get("gaze", {})
            self._head_detector = HeadDetector(
                model_path=self.paths.get_model_path("gaze", "yolov8n_head.onnx"),
                conf_threshold=gaze_config.get("head_conf_th", 0.55),
                head_min_size=gaze_config.get("head_min_size", 20),
                head_max_size=gaze_config.get("head_max_size", 300),
            )
            self._gaze_estimator = GazeEstimator(
                model_path=self.paths.get_model_path("gaze", "gazelle_dinov3_vits16plus_finetuned_1x3x640x640_1xNx4.onnx"),
            )
            roi_json_path = str(self.paths.base_dir / "data" / "ROI.json")
            self._roi_classifier = ROIClassifier(
                roi_json_path=roi_json_path,
                inout_threshold=gaze_config.get("inout_th", 0.5),
                heatmap_threshold=gaze_config.get("heatmap_th", 0.3),
            )
            self._gaze_rois = self._roi_classifier._gaze_rois
            self._head_zones = self._roi_classifier._head_zones
            self._no_gaze_count = 0
            self._NO_GAZE_THRESHOLD = 10
            self._last_alert_ts = -999.0
            self._ALERT_COOLDOWN = 5.0
            # Gaze 缓存（每10帧更新一次，其余帧复用）
            self._cached_gaze_results = []
            self._cached_has_heads = False
            self._cached_any_in_roi = False
            self._pending_alert = False

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
            self._result_storage = MOTResultStorage(self.paths)

            logger.info("MOT 模块初始化完成")
            # 监护制流程状态（通过订阅 bus 事件更新）
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
            kf_dir = self.paths.get_result_dir(self._run_id, "mot") / "key_frames"
            kf_dir.mkdir(parents=True, exist_ok=True)

            role_assigned_done = False

            logger.info(f"开始处理视频: {video_path}")
            logger.info(f"FPS={fps}, 总帧数={total_frames}")

            # 通知前端启动视频流
            self.push_event("video_start", {})

            tracks = []  # 初始化，确保可视化代码始终可用

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                ts = frame_count / fps
                frame_count += 1

                # 更新进度
                self.update_progress(ts, total_frames / fps)

                # 更新聚合器快照（供前端 context 使用）
                self.aggregator.update_module_snapshot("mot", {
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

                    ev = {
                        "localSec": round(ts, 2),
                        "source": "mot",
                        "event": "ROLE_ASSIGNED",
                        "state": "不在监护制",
                        "roles_details": self._roles_info,
                        "frame_id": frame_count,
                    }
                    self._events.append(ev)

                    # 发布到总线
                    self.bus.publish(MsgType.MOT_ROLE_ASSIGNED, {
                        "localSec": round(ts, 2),
                        "roles": self._roles_info,
                    }, ts=ts)

                    # 推送到聚合器（角色分配事件，不含监护状态）
                    self.push_event("tracking", {
                        "localSec": round(ts, 2),
                        "event": "ROLE_ASSIGNED",
                        "roles_details": self._roles_info,
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
                    # 发布到总线
                    self.bus.publish(MsgType.MOT_SUPERVISION_REQUEST, {
                        "localSec": round(ts, 2),
                        "operator": hand_role,
                        "frame_id": frame_count,
                    }, ts=ts)
                    # 记录监护请求事件
                    self._events.append({
                        "localSec": round(ts, 2),
                        "source": "mot",
                        "event": "SUPERVISION_REQUEST",
                        "state": "请求监护",
                        "operator": hand_role,
                        "roles_details": self._roles_info,
                        "frame_id": frame_count,
                    })

                    # 保存关键帧
                    kf_path = kf_dir / f"hand_raise_{hand_role}_{ts:.1f}s.jpg"
                    cv2.imwrite(str(kf_path), frame)

                    ev = {
                        "localSec": round(ts, 2),
                        "source": "mot",
                        "event": "HAND_RAISED",
                        "state": "举手",
                        "operator": hand_role,
                        "key_frame": str(kf_path),
                        "roles_details": self._roles_info,
                    }
                    self._events.append(ev)

                    # 推送到聚合器
                    self.push_event("tracking", {
                        "localSec": round(ts, 2),
                        "event": "HAND_RAISED",
                        "state": "举手",
                        "operator": hand_role,
                        "roles_details": self._roles_info,
                    })

                    logger.info(f"举手检测: {hand_role} @{ts:.1f}s")

                # 距离监控（每30帧检查一次）
                if self._tracker.roles_assigned and frame_count % 30 == 0:
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

                            # 发布距离状态到 bus（供监护制度使用）
                            self.bus.publish(MsgType.MOT_SUPERVISOR_STATUS, {
                                "localSec": round(ts, 2),
                                "state": state_label,
                                "operator": road_name,
                                "distance_px": int(d),
                            }, ts=ts)

                            # 保存事件
                            ev = {
                                "localSec": round(ts, 2),
                                "source": "mot",
                                "event": "SUPERVISOR_STATUS",
                                "state": state_label,
                                "operator": road_name,
                                "distance_px": int(d),
                                "roles_details": self._roles_info,
                            }
                            self._events.append(ev)

                            # 推送到前端同步器（与语音事件对齐显示）
                            self.push_event("tracking", {
                                "localSec": round(ts, 2),
                                "event": "SUPERVISOR_STATUS",
                                "state": state_label,
                                "operator": road_name,
                                "distance_px": int(d),
                                "roles_details": self._roles_info,
                            })

                # 可视化：在帧上画跟踪结果 + 距离 + 注视结果
                if frame_queue is not None and frame_count % 10 == 0:
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
                        _, jpeg = cv2.imencode(".jpg", vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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

        logger.info(f"MOT 结果已保存到 {run_id}")


    def _link_gaze_to_tracks(self, gaze_results, tracks):
        """将注视结果关联到跟踪轨迹，返回 {role: gaze_status}"""
        role_gaze = {}
        if not self._tracker.roles_assigned:
            return role_gaze

        for role_name in ("ROAD1", "ROAD2"):
            track = self._tracker.get_track_by_role(role_name)
            if track is None:
                continue

            tc = track.get_center()
            best_dist = float("inf")
            best_status = None

            for gr in gaze_results:
                gc = np.array(gr["center"])
                d = np.linalg.norm(tc - gc)
                if d < best_dist and d < self._PANEL_MATCH_DIST:
                    best_dist = d
                    best_status = gr["status"]

            role_gaze[role_name] = best_status

        return role_gaze

    def _draw_gaze(self, frame, frame_count, ts, tracks=None):
        """在帧上绘制注视检测结果（ROI、头部框、注视线、告警）
        每10帧跑一次Gaze检测，其余帧复用缓存结果
        """
        import cv2
        import numpy as np

        vis = frame.copy()
        h, w = vis.shape[:2]

        # 画 ROI 区域（半透明）
        if self._gaze_rois:
            overlay = vis.copy()
            for label, contour in self._gaze_rois:
                pts = contour.astype(np.int32).reshape(-1, 2)
                cv2.fillPoly(overlay, [pts], (0, 200, 255))
                cv2.polylines(vis, [pts], True, (0, 200, 255), 2)
                centroid = pts.mean(axis=0).astype(int)
                cv2.putText(vis, label, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

        # 每10帧跑一次Gaze检测，其余帧复用缓存
        GAZE_INTERVAL = 10
        if frame_count % GAZE_INTERVAL == 0:
            # 头部检测
            heads = self._head_detector.detect(frame)
            heads = self._roi_classifier.filter_heads_by_zone(heads)

            # 只对操作员（ROAD1, ROAD2）做注视估计，跳过监护员（LEADER）
            if self._tracker.roles_assigned and tracks and heads:
                operator_heads = []
                for role_name in ("ROAD1", "ROAD2"):
                    track = self._tracker.get_track_by_role(role_name)
                    if track is None:
                        continue
                    tc = track.get_center()
                    # 找到与操作员轨迹最近的头部
                    best_head = None
                    best_dist = float("inf")
                    for head in heads:
                        hc = (head.cx, head.cy)
                        d = ((tc[0] - hc[0])**2 + (tc[1] - hc[1])**2)**0.5
                        if d < best_dist and d < self._PANEL_MATCH_DIST:
                            best_dist = d
                            best_head = head
                    if best_head is not None:
                        operator_heads.append(best_head)
                heads = operator_heads

            self._cached_gaze_results = []
            self._cached_has_heads = bool(heads)
            self._cached_any_in_roi = False

            if heads:
                heatmaps, inout_scores, valid_boxes = self._gaze_estimator.predict(frame, heads)

                if heatmaps is not None and valid_boxes:
                    for i, box in enumerate(valid_boxes):
                        heatmap = heatmaps[i]
                        if heatmap.ndim == 3:
                            heatmap = heatmap[0]

                        inout_score = float(inout_scores[i]) if inout_scores is not None else 1.0
                        gaze_pt = self._roi_classifier.extract_gaze_point(heatmap, w, h)

                        if gaze_pt is None:
                            continue

                        status, roi_label = self._roi_classifier.classify_gaze(inout_score, gaze_pt)

                        if status == "IN_ROI":
                            self._cached_any_in_roi = True

                        self._cached_gaze_results.append({
                            "box": (box.x1, box.y1, box.x2, box.y2),
                            "center": (box.cx, box.cy),
                            "gaze_pt": gaze_pt,
                            "status": status,
                        })

            # 盘台监控：检查 ROAD1, ROAD2 是否在看盘台
            role_gaze = self._link_gaze_to_tracks(self._cached_gaze_results, tracks)
            road1_ok = role_gaze.get("ROAD1") == "IN_ROI"
            road2_ok = role_gaze.get("ROAD2") == "IN_ROI"
            at_least_one = road1_ok or road2_ok

            if not at_least_one:
                if not self._panel_both_away:
                    self._panel_both_away = True
                    self._panel_away_start_ts = ts
                    self._panel_away_start_frame = frame_count
                elif ts - self._panel_away_start_ts >= self._PANEL_AWAY_THRESHOLD:
                    violation = {
                        "start_sec": round(self._panel_away_start_ts, 2),
                        "end_sec": round(ts, 2),
                        "duration_sec": round(ts - self._panel_away_start_ts, 2),
                        "operators": ["ROAD1", "ROAD2"],
                        "frame_start": self._panel_away_start_frame,
                        "frame_end": frame_count,
                    }
                    self._panel_violation_records.append(violation)
                    logger.warning(f"盘台违规: 两人离开{violation['duration_sec']}秒")
                    # 推送盘台违规事件到前端
                    self.push_event("tracking", {
                        "localSec": round(ts, 2),
                        "event": "PANEL_VIOLATION",
                        "source": "mot",
                        "away_duration": violation['duration_sec'],
                    })
                    # 记录盘台违规事件
                    self._events.append({
                        "localSec": round(ts, 2),
                        "source": "mot",
                        "event": "PANEL_VIOLATION",
                        "state": "盘台违规",
                        "away_duration": violation["duration_sec"],
                        "roles_details": self._roles_info,
                    })
                    self._panel_both_away = False
                    self._panel_away_start_ts = -1.0
            else:
                self._panel_both_away = False
                self._panel_away_start_ts = -1.0

            # 告警判断
            if not self._cached_has_heads:
                self._no_gaze_count += 1
                if (self._no_gaze_count >= self._NO_GAZE_THRESHOLD
                        and (ts - self._last_alert_ts) >= self._ALERT_COOLDOWN):
                    self._pending_alert = True
                    self._last_alert_ts = ts
                    self._no_gaze_count = 0
                    # 推送全局告警事件到前端
                    self.push_event("tracking", {
                        "localSec": round(ts, 2),
                        "event": "NO_STAFF_ALERT",
                        "source": "mot",
                    })
                    # 记录无人告警事件
                    self._events.append({
                        "localSec": round(ts, 2),
                        "source": "mot",
                        "event": "NO_STAFF_ALERT",
                        "state": "监控室无人",
                        "roles_details": self._roles_info,
                    })
            else:
                self._no_gaze_count = 0

        # 用缓存结果画图
        color_map = {"IN_ROI": (0, 255, 0), "OUTSIDE_ROI": (0, 0, 255), "OUTSIDE_FRAME": (0, 255, 255)}
        for gr in self._cached_gaze_results:
            color = color_map.get(gr["status"], (255, 255, 255))
            x1, y1, x2, y2 = map(int, gr["box"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            gx, gy = gr["gaze_pt"]
            cx, cy = gr["center"]
            cv2.line(vis, (cx, cy), (gx, gy), color, 2, cv2.LINE_AA)
            cv2.circle(vis, (gx, gy), 4, color, -1)
            cv2.circle(vis, (gx, gy), 6, color, 2)

        # 全局告警条：监控室没人
        if getattr(self, '_pending_alert', False):
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
            cv2.putText(vis, "ALERT: No staff in room!", (w // 2 - 150, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self._pending_alert = False

        # 盘台告警条：两个操作员同时离开盘台
        if self._panel_both_away:
            away_sec = ts - self._panel_away_start_ts
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 45), (w, 85), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
            text = f"PANEL ALERT: Both operators away! {away_sec:.0f}s / 60s"
            cv2.putText(vis, text, (w // 2 - 200, 73),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis
