"""
凝视处理器 — 独立模块，MOT调用.

负责：头部检测、注视推断、ROI分类、可视化、推送推理结果
"""
import cv2
import numpy as np
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List

from modules.gaze.head_detector import HeadDetector
from modules.gaze.gaze_estimator import GazeEstimator
from modules.gaze.roi_classifier import ROIClassifier
from modules.gaze.gaze_attention import GazePoint, GazeAttentionChecker
from modules.gaze.storage_gaze import GazeStorage

logger = logging.getLogger("module.gaze.processor")

# 状态颜色（BGR）
COLOR_MAP = {
    "IN_ROI": (0, 255, 0),  # 绿色
    "OUTSIDE_ROI": (0, 0, 255),  # 红色
    "OUTSIDE_FRAME": (0, 255, 255),  # 黄色
}

class GazeModule:
    """
    凝视处理器.

    独立模块，MOT调用。负责：
    1. 在整帧上做头部检测
    2. 过滤 head_zones 内的头部
    3. 对检测到的头部做凝视估计
    4. 分类注视状态（IN_ROI / OUTSIDE_ROI / OUTSIDE_FRAME）
    5. 在帧上画可视化（ROI、头部框、注视线、注视点、告警条）
    6. 推送推理结果到推理流
    7. 超过60秒告警 → 推送关键事件
    """

    def __init__(
        self,
        head_model_path: str,
        gaze_model_path: str,
        roi_json_path: str,
        config: dict = None,
        inference_fn: Callable = None,
        event_bus=None,
        progress_fn: Callable = None,
        paths=None,
    ):
        """初始化凝视处理器."""
        config = config or {}

        self._event_bus = event_bus
        self._progress_fn = progress_fn
        self._storage = GazeStorage(paths) if paths is not None else None

        # 信息通报流程激活标志
        self._info_notice_active = False

        if self._event_bus is not None:
            try:
                from core.event_bus import EventTopic
                self._event_bus.subscribe(EventTopic.FLOW_STARTED,
                                          self._on_flow_started)
                self._event_bus.subscribe(EventTopic.FLOW_ENDED,
                                          self._on_flow_ended)
                logger.info("GazeModule 已订阅 FLOW_STARTED/FLOW_ENDED 事件")
            except Exception as e:
                logger.warning(f"GazeModule 订阅流程事件失败: {e}")

        self._head_detector = HeadDetector(
            model_path=head_model_path,
            conf_threshold=config.get("head_conf_th", 0.55),
            head_min_size=config.get("head_min_size", 20),
            head_max_size=config.get("head_max_size", 300),
        )

        self._gaze_estimator = GazeEstimator(model_path=gaze_model_path)

        self._roi_classifier = ROIClassifier(
            roi_json_path=roi_json_path,
            inout_threshold=config.get("inout_th", 0.5),
            heatmap_threshold=config.get("heatmap_th", 0.3),
        )

        self._inference_fn = inference_fn

        self._cached_results = []
        self._cached_has_heads = False
        self._cached_any_in_roi = False
        self._gaze_interval = 10

        self._away_start_ts = None
        self._alerting = False
        self._away_threshold = 60.0

        self._events = []
        self._latest_ts = 0.0

        self._attention_checker = GazeAttentionChecker(
            min_turn_displacement=config.get("min_turn_displacement", 100.0),
            min_samples=config.get("min_gaze_samples", 5),
        )
        self._attn_window_start = 0.0
        self._info_notice_start_ts = 0.0
        self._notice_attn_buffer: List[GazePoint] = []
        self._run_id = getattr(paths, "current_run_id",
                               "default") if paths else "default"

        # Gazelle 推理在后台线程执行，不阻塞 tracker 主循环
        self._gaze_executor = ThreadPoolExecutor(max_workers=1,
                                                 thread_name_prefix="gaze")
        self._gaze_future = None
        self._gaze_lock = threading.Lock()  # 保护 _cached_results 等共享状态
        self._last_gaze_ts = 0.0

        logger.info("凝视处理器初始化完成（异步模式）")

    def process_frame(self, frame: np.ndarray, ts: float,
                      frame_count: int) -> np.ndarray:
        """
        处理一帧：检测、估计、可视化、推送.

        异步策略：检测提交到后台线程，主循环不阻塞；未完成则用缓存绘制。
        """
        self._latest_ts = ts
        vis = frame.copy()
        h, w = vis.shape[:2]

        self._draw_rois(vis)
        self._draw_head_zones(vis)

        if frame_count % self._gaze_interval == 0:
            self._try_submit_async_gaze(frame, w, h, ts)

        self._try_collect_async_result()

        # 告警检查在主线程执行，读取缓存状态
        self._check_alert(ts)

        self._update_attention(ts)

        if self._progress_fn:
            self._progress_fn(ts, None)

        with self._gaze_lock:
            self._draw_gaze_results(vis, ts)

        return vis

    def _try_submit_async_gaze(self, frame: np.ndarray, w: int, h: int,
                               ts: float):
        """尝试提交异步凝视检测任务。若上一任务未完成则跳过本轮."""
        if self._gaze_future is not None and not self._gaze_future.done():
            return

        frame_copy = frame.copy()  # 提交副本帧，避免主线程继续修改
        self._gaze_future = self._gaze_executor.submit(
            self._run_gaze_detection_safe, frame_copy, w, h, ts)

    def _try_collect_async_result(self):
        """检查后台任务是否完成。完成则刷新缓存（_run_gaze_detection_safe 内部已处理）."""
        if self._gaze_future is not None and self._gaze_future.done():
            exc = self._gaze_future.exception()  # 防止异常被静默吞掉
            if exc is not None:
                logger.warning(f"异步凝视检测失败: {exc}")
            self._gaze_future = None

    def _run_gaze_detection_safe(self, frame: np.ndarray, w: int, h: int,
                                 ts: float):
        """_run_gaze_detection 的线程安全包装：加锁保护缓存写入."""
        try:
            self._run_gaze_detection(frame, w, h, ts)
        except Exception as e:
            logger.warning(f"凝视检测异常: {e}", exc_info=True)

    def _run_gaze_detection(self, frame: np.ndarray, w: int, h: int,
                            ts: float):
        """运行凝视检测（在后台线程执行，通过 _gaze_lock 保护共享缓存."""
        all_heads = self._head_detector.detect(frame)
        heads = self._roi_classifier.filter_heads_by_zone(all_heads)

        # 先在局部变量构建结果，再一次性加锁写缓存，减少锁持有时间
        new_results = []
        new_has_heads = bool(heads)
        new_any_in_roi = False

        if heads:
            heatmaps, inout_scores, valid_boxes = self._gaze_estimator.predict(
                frame, heads)
            if heatmaps is not None and valid_boxes:
                for i, box in enumerate(valid_boxes):
                    heatmap = heatmaps[i]
                    if heatmap.ndim == 3:
                        heatmap = heatmap[0]
                    inout_score = float(
                        inout_scores[i]) if inout_scores is not None else 1.0
                    gaze_pt = self._roi_classifier.extract_gaze_point(
                        heatmap, w, h)
                    if gaze_pt is None:
                        continue
                    status, roi_label = self._roi_classifier.classify_gaze(
                        inout_score, gaze_pt)
                    if status == "IN_ROI":
                        new_any_in_roi = True
                    new_results.append({
                        "box": (box.x1, box.y1, box.x2, box.y2),
                        "center": (box.cx, box.cy),
                        "gaze_pt": gaze_pt,
                        "status": status,
                    })

        with self._gaze_lock:
            self._cached_results = new_results
            self._cached_has_heads = new_has_heads
            self._cached_any_in_roi = new_any_in_roi
            self._last_gaze_ts = ts

        if self._inference_fn:
            away_dur = 0.0
            if (new_has_heads and not new_any_in_roi
                    and self._away_start_ts is not None):
                away_dur = ts - self._away_start_ts
            self._inference_fn(
                "gaze", {
                    "localSec": round(ts, 2),
                    "tag": "gaze_status",
                    "data": {
                        "has_heads": new_has_heads,
                        "any_in_roi": new_any_in_roi,
                        "heads_count": len(new_results),
                        "away_duration": round(away_dur, 2),
                    },
                })

        # 告警逻辑（_check_alert）移到主线程 process_frame 中执行，避免跨线程竞争 _away_start_ts / _alerting

    def _check_alert(self, ts: float):
        """检查告警条件（主线程调用，读取缓存需加锁."""
        with self._gaze_lock:
            has_heads = self._cached_has_heads
            any_in_roi = self._cached_any_in_roi

        if has_heads and not any_in_roi:
            if self._away_start_ts is None:
                self._away_start_ts = ts
            away_dur = ts - self._away_start_ts
            if away_dur >= self._away_threshold and not self._alerting:
                self._alerting = True
                # 推理流 data 只含纯展示字段
                if self._inference_fn:
                    self._inference_fn(
                        "gaze", {
                            "localSec": round(ts, 2),
                            "tag": "GAZE_ALERT",
                            "data": {
                                "state": "无人注视盘台",
                                "away_duration": round(away_dur, 2),
                            },
                        })
                # 事件流：完整字段供规则状态机使用
                if self._event_bus:
                    from core.event_bus import EventTopic
                    with self._gaze_lock:
                        heads_count = len(self._cached_results)
                    self._event_bus.publish(EventTopic.GAZE_ALERT, {
                        "localSec": round(ts, 2),
                        "state": "无人注视盘台",
                        "away_duration": round(away_dur, 2),
                        "heads_count": heads_count,
                    },
                                            ts=ts)
                logger.warning(f"凝视告警: 无人注视盘台 {away_dur:.1f}秒 @{ts:.1f}s")
        else:
            if self._alerting:
                duration = (ts - self._away_start_ts
                            if self._away_start_ts else 0.0)
                self._events.append({
                    "localSec":
                    round(self._away_start_ts, 2),
                    "key_moment":
                    f"没有看盘台持续{round(duration, 1)}秒",
                })
                if self._storage:
                    self._storage.save_key_moments(self._run_id, self.get_events())
                # 推理流：通知前端告警结束
                if self._inference_fn:
                    self._inference_fn(
                        "gaze", {
                            "localSec": round(ts, 2),
                            "tag": "GAZE_VIOLATION_END",
                            "data": {
                                "state": "无人注视盘台",
                                "duration": round(duration, 2),
                            },
                        })
                # 事件流：完整字段供规则状态机使用
                if self._event_bus:
                    from core.event_bus import EventTopic
                    self._event_bus.publish(EventTopic.GAZE_ALERT, {
                        "localSec": round(ts, 2),
                        "state": "violation_end",
                        "duration": round(duration, 2),
                    },
                                            ts=ts)
            self._away_start_ts = None
            self._alerting = False

    def _on_flow_started(self, msg: dict) -> None:
        """流程开始事件回调：信息通报流程激活时，记录触发时间并开启 10S 关注度检测窗口."""
        data = msg.get("data", {})
        ts = data.get("localSec", msg.get("ts", self._latest_ts))
        if data.get("flow_type") == "info_notice":
            self._info_notice_start_ts = ts
            self._info_notice_active = True
            self._notice_attn_buffer = []
            logger.info(f"GazeModule: 收到信息通报触发，开启 10 秒关注度检测窗口 @{ts:.1f}s")

    def _on_flow_ended(self, msg: dict) -> None:
        """流程结束事件回调：信息通报流程结束时，关闭 ATTENTION_RESULT 推送."""
        data = msg.get("data", {})
        if data.get("flow_type") == "info_notice":
            self._info_notice_active = False
            logger.info("GazeModule: 信息通报流程结束")

    def _update_attention(self, ts: float):
        """信息通报流程触发后 10 秒关注度评估与 keymoment 保存."""
        if not self._info_notice_active:
            return

        if ts <= self._info_notice_start_ts + 10.0:
            with self._gaze_lock:
                cached_snapshot = list(self._cached_results)
            if cached_snapshot:
                gx_mean = sum(r["gaze_pt"][0]
                              for r in cached_snapshot) / len(cached_snapshot)
                gy_mean = sum(r["gaze_pt"][1]
                              for r in cached_snapshot) / len(cached_snapshot)
                self._notice_attn_buffer.append(
                    GazePoint(ts * 1000.0, gx_mean, gy_mean))
        else:
            has_turned = False
            if self._notice_attn_buffer:
                result = self._attention_checker.evaluate(
                    self._notice_attn_buffer)
                has_turned = result.has_turned

            key_moment_str = "已给予关注" if has_turned else "没有给予关注"
            record = {
                "localSec": round(self._info_notice_start_ts, 2),
                "key_moment": key_moment_str,
            }
            self._events.append(record)

            if self._storage:
                self._storage.save_key_moments(self._run_id, self.get_events())

            if self._event_bus:
                from core.event_bus import EventTopic
                self._event_bus.publish(EventTopic.GAZE_ATTENTION, {
                    "localSec": round(ts, 2),
                    "has_turned": has_turned,
                },
                                        ts=ts)

            logger.info(
                f"GazeModule: 信息通报 10S 关注度评估完成 @{ts:.1f}s 结果='{key_moment_str}'"
            )
            self._info_notice_active = False
            self._notice_attn_buffer = []

    def _draw_rois(self, vis: np.ndarray):
        """绘制rois."""
        rois = self._roi_classifier._gaze_rois
        if not rois:
            return
        overlay = vis.copy()
        for label, contour in rois:
            pts = contour.astype(np.int32).reshape(-1, 2)
            cv2.fillPoly(overlay, [pts], (0, 200, 255))
            cv2.polylines(vis, [pts], True, (0, 200, 255), 2)
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(vis, label, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 255), 2)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

    def _draw_head_zones(self, vis: np.ndarray):
        """绘制头部zones."""
        zones = self._roi_classifier._head_zones
        if not zones:
            return
        for label, contour in zones:
            pts = contour.astype(np.int32).reshape(-1, 2)
            for j in range(len(pts)):
                if j % 2 == 0:
                    cv2.line(vis, tuple(pts[j]),
                             tuple(pts[(j + 1) % len(pts)]), (0, 255, 0), 1,
                             cv2.LINE_AA)

    def _draw_gaze_results(self, vis: np.ndarray, ts: float = 0.0):
        """绘制注视results."""
        for gr in self._cached_results:
            color = COLOR_MAP.get(gr["status"], (255, 255, 255))
            x1, y1, x2, y2 = map(int, gr["box"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            gx, gy = gr["gaze_pt"]
            cx, cy = gr["center"]
            cv2.line(vis, (cx, cy), (gx, gy), color, 2, cv2.LINE_AA)
            cv2.circle(vis, (gx, gy), 4, color, -1)
            cv2.circle(vis, (gx, gy), 6, color, 2)

        pass  # 前端/画面渲染 Attended 标签已完全移除

    def get_events(self) -> list:
        """获取events."""
        events = list(self._events)
        # 视频结束时若仍处于告警状态，按当前帧计算持续时间并追加
        if self._alerting and self._away_start_ts is not None:
            duration = self._latest_ts - self._away_start_ts
            events.append({
                "localSec": round(self._away_start_ts, 2),
                "key_moment": f"没有看盘台持续{round(duration, 1)}秒",
            })
        return events

    def save_results(self, run_id: str) -> None:
        """保存results."""
        self._flush_async()
        self._run_id = run_id
        if self._storage:
            self._storage.save_key_moments(run_id, self.get_events())

    def _flush_async(self):
        """刷出异步."""
        if self._gaze_future is not None:
            try:
                self._gaze_future.result(timeout=5.0)
            except Exception as e:
                logger.warning(f"等待异步凝视检测完成时异常: {e}")
            self._gaze_future = None

    def shutdown(self):
        """shutdown."""
        self._flush_async()
        self._gaze_executor.shutdown(wait=False)
