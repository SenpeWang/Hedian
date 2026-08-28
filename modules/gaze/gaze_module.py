"""凝视模块 — 独立模块，由 MOT 主循环调用.

负责头部检测、注视推断、ROI 分类与可视化，并推送推理结果与注视告警事件。
"""
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from core.path_manager import PathConfig

from modules.gaze.gaze_attention import GazeAttentionChecker, GazePoint
from modules.gaze.gaze_estimator import GazeEstimator
from modules.gaze.head_detector import HeadDetector
from modules.gaze.roi_classifier import ROIClassifier
from modules.gaze.storage_gaze import GazeStorage

logger = logging.getLogger("module.gaze")

# 状态颜色（BGR）
COLOR_MAP = {
    "IN_ROI": (0, 255, 0),  # 绿色
    "OUTSIDE_ROI": (0, 0, 255),  # 红色
    "OUTSIDE_FRAME": (0, 255, 255),  # 黄色
}


class GazeModule:
    """凝视模块.

    独立模块，由 MOT 主循环调用。负责：
    1. 在整帧上做头部检测
    2. 只保留头部区域多边形（head_zones）内的头部
    3. 对检测到的头部做凝视估计
    4. 分类注视状态（IN_ROI / OUTSIDE_ROI / OUTSIDE_FRAME）
    5. 在帧上画可视化（ROI、头部框、注视线、注视点、告警条）
    6. 推送推理结果到推理流
    7. 超过 60 秒告警 → 推送关键事件
    """

    def __init__(
        self,
        head_model_path: str,
        gaze_model_path: str,
        roi_json_path: str,
        config: Optional[Dict[str, Any]] = None,
        inference_fn: Optional[Callable] = None,
        event_bus: Optional[Any] = None,
        progress_fn: Optional[Callable] = None,
        paths: Optional[PathConfig] = None,
    ) -> None:
        """初始化凝视模块.

        Args:
            head_model_path: YOLOv8 头部检测模型路径。
            gaze_model_path: Gazelle 注视推断模型路径。
            roi_json_path: ROI 配置文件路径。
            config: 凝视模块配置字典；None 时全部使用默认值。
            inference_fn: 推理结果推送回调。
            event_bus: 事件总线；None 时不订阅也不发布事件。
            progress_fn: 进度回调。
            paths: 路径配置；None 时不落盘关键事件。
        """
        config = config or {}

        self._event_bus = event_bus
        self._progress_fn = progress_fn
        self._storage = GazeStorage(paths) if paths is not None else None

        # 信息通报流程激活标志
        self._info_notice_active = False

        if self._event_bus is not None:
            from core.event_bus import EventTopic
            self._event_bus.subscribe(EventTopic.FLOW_STARTED,
                                      self._on_flow_started)
            self._event_bus.subscribe(EventTopic.FLOW_ENDED, self._on_flow_ended)
            logger.info("GazeModule 已订阅 FLOW_STARTED/FLOW_ENDED 事件")

        self._head_detector = HeadDetector(
            model_path=head_model_path,
            conf_threshold=config.get("head_conf_th", 0.55),
            head_min_size=config.get("head_min_size", 20),
            head_max_size=config.get("head_max_size", 300),
        )

        self._gaze_estimator = GazeEstimator(model_path=gaze_model_path)

        self._roi_classifier = ROIClassifier(
            roi_json_path=roi_json_path,
            in_out_threshold=config.get("inout_th", 0.5),
            heatmap_threshold=config.get("heatmap_th", 0.3),
        )

        self._inference_fn = inference_fn

        self._cached_results: List[Dict[str, Any]] = []
        self._cached_has_heads = False
        self._cached_any_in_roi = False
        self._gaze_interval = int(config.get("gaze_interval", 10)) if config else 10

        self._gaze_away_start_timestamp: Optional[float] = None
        self._alerting = False
        self._gaze_away_threshold = 60.0

        self._events: List[Dict[str, Any]] = []
        self._latest_timestamp = 0.0

        self._attention_checker = GazeAttentionChecker(
            min_turn_displacement=config.get("min_turn_displacement", 100.0),
            min_samples=config.get("min_gaze_samples", 5),
        )
        self._attention_window_start = 0.0
        self._info_notice_start_timestamp = 0.0
        self._notice_attention_buffer: List[GazePoint] = []
        self._run_id = (
            getattr(paths, "current_run_id", "default") if paths else "default"
        )

        # Gazelle 推理在后台线程执行，不阻塞 tracker 主循环
        self._gaze_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gaze"
        )
        self._gaze_future: Optional[Future[Any]] = None
        self._gaze_lock = threading.Lock()  # 保护 _cached_results 等共享状态
        self._last_gaze_timestamp = 0.0

        logger.info("凝视处理器初始化完成（异步模式）")

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_count: int,
    ) -> np.ndarray:
        """处理一帧：检测、估计、可视化、推送.

        异步策略：检测提交到后台线程，主循环不阻塞；未完成则用缓存绘制。

        Args:
            frame: 原始 BGR 帧。
            timestamp: 帧时间戳（秒）。
            frame_count: 帧序号。

        Returns:
            叠加可视化结果的帧副本。
        """
        self._latest_timestamp = timestamp
        annotated_frame = frame.copy()
        frame_height, frame_width = annotated_frame.shape[:2]

        self._draw_rois(annotated_frame)
        self._draw_head_zones(annotated_frame)

        if frame_count % self._gaze_interval == 0:
            self._try_submit_async_gaze(
                frame, frame_width, frame_height, timestamp
            )

        self._try_collect_async_result()

        # 告警检查在主线程执行，读取缓存状态
        self._check_alert(timestamp)

        self._update_attention(timestamp)

        if self._progress_fn:
            self._progress_fn(timestamp, None)

        with self._gaze_lock:
            self._draw_gaze_results(annotated_frame, timestamp)

        return annotated_frame

    def _try_submit_async_gaze(
        self,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
        timestamp: float,
    ) -> None:
        """尝试提交异步凝视检测任务。若上一任务未完成则跳过本轮."""
        if self._gaze_future is not None and not self._gaze_future.done():
            return

        frame_copy = frame.copy()  # 提交副本帧，避免主线程继续修改
        self._gaze_future = self._gaze_executor.submit(
            self._run_gaze_detection_safe,
            frame_copy,
            frame_width,
            frame_height,
            timestamp,
        )

    def _try_collect_async_result(self) -> None:
        """检查后台任务是否完成。完成则刷新缓存（_run_gaze_detection_safe 内部已处理）."""
        if self._gaze_future is not None and self._gaze_future.done():
            exception = self._gaze_future.exception()  # 防止异常被静默吞掉
            if exception is not None:
                logger.warning(f"异步凝视检测失败: {exception}")
            self._gaze_future = None

    def _run_gaze_detection_safe(
        self,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
        timestamp: float,
    ) -> None:
        """_run_gaze_detection 的线程安全包装：加锁保护缓存写入."""
        try:
            self._run_gaze_detection(
                frame, frame_width, frame_height, timestamp
            )
        except Exception as error:
            logger.warning(f"凝视检测异常: {error}", exc_info=True)

    def _run_gaze_detection(
        self,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
        timestamp: float,
    ) -> None:
        """运行凝视检测（在后台线程执行，通过 _gaze_lock 保护共享缓存）.

        Args:
            frame: 原始 BGR 帧。
            frame_width: 帧宽度。
            frame_height: 帧高度。
            timestamp: 帧时间戳（秒）。
        """
        all_heads = self._head_detector.detect(frame)
        heads = self._roi_classifier.filter_heads_by_zone(all_heads)

        # 先在局部变量构建结果，再一次性加锁写缓存，减少锁持有时间
        new_results: List[Dict[str, Any]] = []
        new_has_heads = bool(heads)
        new_any_in_roi = False

        if heads:
            heatmaps, in_out_scores, valid_boxes = self._gaze_estimator.predict(
                frame, heads
            )
            if heatmaps is not None and valid_boxes:
                for index, box in enumerate(valid_boxes):
                    heatmap = heatmaps[index]
                    if heatmap.ndim == 3:
                        heatmap = heatmap[0]
                    in_out_score = (
                        float(in_out_scores[index])
                        if in_out_scores is not None else 1.0
                    )
                    gaze_point = self._roi_classifier.extract_gaze_point(
                        heatmap, frame_width, frame_height
                    )
                    if gaze_point is None:
                        continue
                    status, roi_label = self._roi_classifier.classify_gaze(
                        in_out_score, gaze_point
                    )
                    if status == "IN_ROI":
                        new_any_in_roi = True
                    new_results.append({
                        "box": (box.x1, box.y1, box.x2, box.y2),
                        "center": (box.cx, box.cy),
                        "gaze_pt": gaze_point,
                        "status": status,
                    })

        with self._gaze_lock:
            self._cached_results = new_results
            self._cached_has_heads = new_has_heads
            self._cached_any_in_roi = new_any_in_roi
            self._last_gaze_timestamp = timestamp

        if self._inference_fn:
            away_duration = 0.0
            if (new_has_heads and not new_any_in_roi
                    and self._gaze_away_start_timestamp is not None):
                away_duration = timestamp - self._gaze_away_start_timestamp
            self._inference_fn(
                "gaze", {
                    "localSec": round(timestamp, 2),
                    "tag": "gaze_status",
                    "data": {
                        "has_heads": new_has_heads,
                        "any_in_roi": new_any_in_roi,
                        "heads_count": len(new_results),
                        "away_duration": round(away_duration, 2),
                    },
                }
            )

        # 告警逻辑（_check_alert）移到主线程 process_frame 中执行，
        # 避免跨线程竞争 _gaze_away_start_timestamp / _alerting

    def _check_alert(self, timestamp: float) -> None:
        """检查告警条件（主线程调用，读取缓存需加锁）.

        Args:
            timestamp: 帧时间戳（秒）。
        """
        with self._gaze_lock:
            has_heads = self._cached_has_heads
            any_in_roi = self._cached_any_in_roi

        if has_heads and not any_in_roi:
            if self._gaze_away_start_timestamp is None:
                self._gaze_away_start_timestamp = timestamp
            away_duration = timestamp - self._gaze_away_start_timestamp
            if away_duration >= self._gaze_away_threshold and not self._alerting:
                self._alerting = True
                # 推理流 data 只含纯展示字段
                if self._inference_fn:
                    self._inference_fn(
                        "gaze", {
                            "localSec": round(timestamp, 2),
                            "tag": "GAZE_ALERT",
                            "data": {
                                "state": "无人注视盘台",
                                "away_duration": round(away_duration, 2),
                            },
                        }
                    )
                # 事件流：完整字段供规则状态机使用
                if self._event_bus:
                    from core.event_bus import EventTopic
                    with self._gaze_lock:
                        heads_count = len(self._cached_results)
                    self._event_bus.publish(
                        EventTopic.GAZE_ALERT,
                        {
                            "localSec": round(timestamp, 2),
                            "state": "无人注视盘台",
                            "away_duration": round(away_duration, 2),
                            "heads_count": heads_count,
                        },
                        timestamp=timestamp,
                    )
                logger.warning(
                    f"凝视告警: 无人注视盘台 {away_duration:.1f}秒 @{timestamp:.1f}s"
                )
        else:
            if self._alerting:
                duration = (
                    timestamp - self._gaze_away_start_timestamp
                    if self._gaze_away_start_timestamp else 0.0
                )
                self._events.append({
                    "localSec": round(self._gaze_away_start_timestamp, 2),
                    "key_moment": f"没有看盘台持续{round(duration, 1)}秒",
                })
                if self._storage:
                    self._storage.save_key_moments(self._run_id, self.get_events())
                # 推理流：通知前端告警结束
                if self._inference_fn:
                    self._inference_fn(
                        "gaze", {
                            "localSec": round(timestamp, 2),
                            "tag": "GAZE_VIOLATION_END",
                            "data": {
                                "state": "无人注视盘台",
                                "duration": round(duration, 2),
                            },
                        }
                    )
                # 事件流：完整字段供规则状态机使用
                if self._event_bus:
                    from core.event_bus import EventTopic
                    self._event_bus.publish(
                        EventTopic.GAZE_ALERT,
                        {
                            "localSec": round(timestamp, 2),
                            "state": "violation_end",
                            "duration": round(duration, 2),
                        },
                        timestamp=timestamp,
                    )
            self._gaze_away_start_timestamp = None
            self._alerting = False

    def _on_flow_started(self, event: dict) -> None:
        """流程开始事件回调：信息通报流程激活时，记录触发时间并开启 10S 关注度检测窗口.

        Args:
            event: 事件信封；载荷位于 data 键。
        """
        payload = event.get("data", {})
        timestamp = payload.get(
            "localSec", event.get("ts", self._latest_timestamp)
        )
        if payload.get("flow_type") == "info_notice":
            self._info_notice_start_timestamp = timestamp
            self._info_notice_active = True
            self._notice_attention_buffer = []
            logger.info(f"GazeModule: 收到信息通报触发，开启 10 秒关注度检测窗口 @{timestamp:.1f}s")

    def _on_flow_ended(self, event: dict) -> None:
        """流程结束事件回调：信息通报流程结束时，关闭 ATTENTION_RESULT 推送.

        Args:
            event: 事件信封；载荷位于 data 键。
        """
        payload = event.get("data", {})
        if payload.get("flow_type") == "info_notice":
            self._info_notice_active = False
            logger.info("GazeModule: 信息通报流程结束")

    def _update_attention(self, timestamp: float) -> None:
        """信息通报流程触发后 10 秒关注度评估与 keymoment 保存.

        Args:
            timestamp: 帧时间戳（秒）。
        """
        if not self._info_notice_active:
            return

        if timestamp <= self._info_notice_start_timestamp + 10.0:
            with self._gaze_lock:
                cached_snapshot = list(self._cached_results)
            if cached_snapshot:
                mean_point_x = sum(
                    result["gaze_pt"][0] for result in cached_snapshot
                ) / len(cached_snapshot)
                mean_point_y = sum(
                    result["gaze_pt"][1] for result in cached_snapshot
                ) / len(cached_snapshot)
                self._notice_attention_buffer.append(
                    GazePoint(
                        timestamp_ms=timestamp * 1000.0,
                        x=mean_point_x,
                        y=mean_point_y,
                    )
                )
        else:
            has_turned = False
            if self._notice_attention_buffer:
                attention_result = self._attention_checker.evaluate(
                    self._notice_attention_buffer
                )
                has_turned = attention_result.has_turned

            key_moment = "已给予关注" if has_turned else "没有给予关注"
            record = {
                "localSec": round(self._info_notice_start_timestamp, 2),
                "key_moment": key_moment,
            }
            self._events.append(record)

            if self._storage:
                self._storage.save_key_moments(self._run_id, self.get_events())

            if self._event_bus:
                from core.event_bus import EventTopic
                self._event_bus.publish(
                    EventTopic.GAZE_ATTENTION,
                    {
                        "localSec": round(timestamp, 2),
                        "has_turned": has_turned,
                    },
                    timestamp=timestamp,
                )

            logger.info(
                f"GazeModule: 信息通报 10S 关注度评估完成 @{timestamp:.1f}s "
                f"结果='{key_moment}'"
            )
            self._info_notice_active = False
            self._notice_attention_buffer = []

    def _draw_rois(self, annotated_frame: np.ndarray) -> None:
        """在帧上绘制 ROI 多边形与标签.

        Args:
            annotated_frame: 待叠加可视化的帧。
        """
        rois = self._roi_classifier._gaze_rois
        if not rois:
            return
        overlay = annotated_frame.copy()
        for label, contour in rois:
            vertices = contour.astype(np.int32).reshape(-1, 2)
            cv2.fillPoly(overlay, [vertices], (0, 200, 255))
            cv2.polylines(annotated_frame, [vertices], True, (0, 200, 255), 2)
            centroid = vertices.mean(axis=0).astype(int)
            cv2.putText(
                annotated_frame,
                label,
                tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )
        cv2.addWeighted(overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)

    def _draw_head_zones(self, annotated_frame: np.ndarray) -> None:
        """在帧上绘制头部区域多边形.

        Args:
            annotated_frame: 待叠加可视化的帧。
        """
        zones = self._roi_classifier._head_zones
        if not zones:
            return
        for label, contour in zones:
            vertices = contour.astype(np.int32).reshape(-1, 2)
            for line_index in range(len(vertices)):
                if line_index % 2 == 0:
                    cv2.line(
                        annotated_frame,
                        tuple(vertices[line_index]),
                        tuple(vertices[(line_index + 1) % len(vertices)]),
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

    def _draw_gaze_results(
        self,
        annotated_frame: np.ndarray,
        timestamp: float = 0.0,
    ) -> None:
        """在帧上绘制头部框、注视线与注视点.

        Args:
            annotated_frame: 待叠加可视化的帧。
            timestamp: 帧时间戳（秒），保留参数以对齐调用方签名。
        """
        for gaze_result in self._cached_results:
            color = COLOR_MAP.get(gaze_result["status"], (255, 255, 255))
            x1, y1, x2, y2 = map(int, gaze_result["box"])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            gaze_x, gaze_y = gaze_result["gaze_pt"]
            center_x, center_y = gaze_result["center"]
            cv2.line(
                annotated_frame,
                (center_x, center_y),
                (gaze_x, gaze_y),
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(annotated_frame, (gaze_x, gaze_y), 4, color, -1)
            cv2.circle(annotated_frame, (gaze_x, gaze_y), 6, color, 2)

    def get_events(self) -> List[Dict[str, Any]]:
        """获取关键事件列表.

        Returns:
            关键事件列表副本。
        """
        events = list(self._events)
        # 视频结束时若仍处于告警状态，按当前帧计算持续时间并追加
        if self._alerting and self._gaze_away_start_timestamp is not None:
            duration = self._latest_timestamp - self._gaze_away_start_timestamp
            events.append({
                "localSec": round(self._gaze_away_start_timestamp, 2),
                "key_moment": f"没有看盘台持续{round(duration, 1)}秒",
            })
        return events

    def save_results(self, run_id: str) -> None:
        """刷新后台任务并把关键事件落盘.

        Args:
            run_id: 本次运行标识。
        """
        self._flush_async()
        self._run_id = run_id
        if self._storage:
            self._storage.save_key_moments(run_id, self.get_events())

    def _flush_async(self) -> None:
        """等待后台凝视检测任务完成."""
        if self._gaze_future is not None:
            try:
                self._gaze_future.result(timeout=5.0)
            except Exception as error:
                logger.warning(f"等待异步凝视检测完成时异常: {error}")
            self._gaze_future = None

    def shutdown(self) -> None:
        """刷新后台任务并关闭凝视线程池."""
        self._flush_async()
        self._gaze_executor.shutdown(wait=False)
