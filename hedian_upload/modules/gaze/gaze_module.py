"""凝视模块 — 独立模块，由 MOT 主循环调用.

负责头部检测、注视推断、ROI 分类与可视化，并推送推理结果与注视告警事件。
"""
import logging
import math
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


# 工位身份基线坐标（与 modules/tracker/multi_object_tracker.WORKSTATIONS 对齐）：
# 按头部中心到各基线的欧氏距离做最近邻匹配，距离过大则记为未归属
# (视频中 LEADER/ROAD1/ROAD2 操作员头部长期停留位置)
OPERATOR_BASELINES: Dict[str, tuple] = {
    "LEADER": (1049, 398),
    "ROAD1":  (1563, 494),
    "ROAD2":  (1146, 662),
}

# 最近邻匹配最大距离（像素）：超出此值视为头部漂出工位，记为 None 避免误归属
OPERATOR_MATCH_MAX_DIST_PX = 280


def _assign_operator(head_cx: float, head_cy: float) -> Optional[str]:
    """根据头部中心坐标匹配工位基线，返回 LEADER/ROAD1/ROAD2 或 None.

    Args:
        head_cx: 头部检测框中心 x 坐标（像素）.
        head_cy: 头部检测框中心 y 坐标（像素）.

    Returns:
        最近的工位身份；所有基线距离均超过 OPERATOR_MATCH_MAX_DIST_PX 时返回 None.
    """
    best_name: Optional[str] = None
    best_dist = float("inf")
    for name, (bx, by) in OPERATOR_BASELINES.items():
        dx = head_cx - bx
        dy = head_cy - by
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    if best_dist > OPERATOR_MATCH_MAX_DIST_PX:
        return None
    return best_name


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

    #: 信息通报流程触发后的关注度采样窗口长度（秒）：窗口按画面时间戳推进，不按事件到达时刻
    NOTICE_ATTENTION_WINDOW_SEC = 10.0

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
        run_id: str = "",
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
            run_id: 本次运行标识；由宿主 TrackerModule 注入，留空时中途落盘会被跳过。
        """
        config = config or {}

        self._event_bus = event_bus
        self._progress_fn = progress_fn
        self._storage = GazeStorage(paths) if paths is not None else None

        # 信息通报关注度窗口：flow_id -> {"start": 起始秒, "buffer": {操作员: [GazePoint]}}。
        # 按流程号分别登记与采样：voice 远快于逐帧推理，多个流程的窗口会在画面时间轴上
        # 前后紧邻地开启（甚至后一个触发先到），单窗口实现会让后到的触发清空先开的采样，
        # 使先发起的流程恒判"未关注"。
        self._notice_windows: Dict[int, Dict[str, Any]] = {}

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
        # 运行标识: 由宿主 TrackerModule 在 initialize 时注入. 早期实现回退到 "default"
        # (PathConfig 并无 current_run_id 属性), 使流程中途落盘的关注度结论被写到
        # data/results/default/gaze/ 下, 评估层按本次 run_id 取证时永远读不到 (gaze 恒 0 条).
        self._run_id = run_id

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
                        "operator": _assign_operator(box.cx, box.cy),
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
                if self._storage and self._run_id:
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
        """流程开始事件回调：为本次信息通报流程登记 10S 关注度检测窗口.

        按 flow_id 分别登记，各窗口独立采样、互不覆盖。

        Args:
            event: 事件信封；载荷位于 data 键。
        """
        payload = event.get("data", {})
        timestamp = payload.get(
            "localSec", event.get("ts", self._latest_timestamp)
        )
        if payload.get("flow_type") == "info_notice":
            flow_id = payload.get("flow_id")
            flow_id = int(flow_id) if flow_id is not None else -1
            self._notice_windows[flow_id] = {
                "start": float(timestamp), "buffer": {}}
            logger.info(
                f"GazeModule: 收到信息通报触发 flow_id={flow_id}，开启 "
                f"{self.NOTICE_ATTENTION_WINDOW_SEC:.0f} 秒"
                f"关注度检测窗口 @{timestamp:.1f}s")

    def _on_flow_ended(self, event: dict) -> None:
        """流程结束事件回调：仅记录流程结束，关注度窗口按画面时间戳自然收口.

        语音事件远快于逐帧推理（本机约 3 倍速），FLOW_ENDED 常在画面尚未走到关注度窗口
        末端时就已到达；此处若直接停表，10 秒关注度结论将永远无法产出，评估层只能恒判
        "无关注"。窗口收口改由 _update_attention 在帧时间戳越过窗口末端时完成；若视频
        提前结束，由 save_results 用已采样样本兜底收口。

        Args:
            event: 事件信封；载荷位于 data 键。
        """
        payload = event.get("data", {})
        if payload.get("flow_type") == "info_notice":
            logger.info("GazeModule: 信息通报流程结束（关注度窗口按画面时间戳收口）")

    def _update_attention(self, timestamp: float) -> None:
        """推进各流程的关注度窗口：窗口内采样、越过窗口末端即收口（逐帧调用）.

        窗口按画面时间戳推进，不按事件到达时刻：落在某一窗口区间内的帧把当前注视点
        按操作员归并进该窗口的缓冲区，首个越过窗口末端的帧触发该窗口收口。

        Args:
            timestamp: 帧时间戳（秒）。
        """
        if not self._notice_windows:
            return

        with self._gaze_lock:
            cached_snapshot = list(self._cached_results)

        for flow_id in list(self._notice_windows.keys()):
            window = self._notice_windows.get(flow_id)
            if window is None:
                continue
            start = window["start"]
            if timestamp > start + self.NOTICE_ATTENTION_WINDOW_SEC:
                self._conclude_attention(flow_id, timestamp)
                continue
            # 窗口起点之前的帧不参与判定：语音事件领先画面推理数秒，
            # 若把流程触发前的历史注视点也算进缓冲区，会稀释"是否转头"的判定
            if timestamp < start:
                continue
            per_op: Dict[str, List[tuple]] = {}
            for result in cached_snapshot:
                op = result.get("operator")
                if op:
                    per_op.setdefault(op, []).append(result["gaze_pt"])
            for op, pts in per_op.items():
                mx = sum(p[0] for p in pts) / len(pts)
                my = sum(p[1] for p in pts) / len(pts)
                window["buffer"].setdefault(op, []).append(
                    GazePoint(timestamp_ms=timestamp * 1000.0, x=mx, y=my)
                )

    def _conclude_attention(self, flow_id: int, timestamp: float) -> None:
        """收口指定流程的关注度窗口：判定结果并落盘 keymoment、发布 GAZE_ATTENTION.

        Args:
            flow_id: 流程号，与 FLOW_STARTED 的 flow_id 一致；规则层据此把结论
                归属到对应流程（被覆盖收口的流程也能拿到自己的迟到结论）。
            timestamp: 触发收口的帧时间戳（秒），用于事件时间与日志。
        """
        window = self._notice_windows.pop(flow_id, None)
        if window is None:
            return

        operators_turned: Dict[str, bool] = {}
        for op, pts in window["buffer"].items():
            if pts:
                operators_turned[op] = self._attention_checker.evaluate(pts).has_turned
        has_turned = any(operators_turned.values())

        key_moment = "已给予关注" if has_turned else "没有给予关注"
        record = {
            "localSec": round(window["start"], 2),
            "key_moment": key_moment,
            "operators": operators_turned,
        }
        self._events.append(record)

        if self._storage and self._run_id:
            self._storage.save_key_moments(self._run_id, self.get_events())

        if self._event_bus:
            from core.event_bus import EventTopic
            self._event_bus.publish(
                EventTopic.GAZE_ATTENTION,
                {
                    "flow_id": flow_id,
                    "localSec": round(timestamp, 2),
                    "has_turned": has_turned,
                    "operators": operators_turned,
                },
                timestamp=timestamp,
            )

        logger.info(
            f"GazeModule: 信息通报 {self.NOTICE_ATTENTION_WINDOW_SEC:.0f}S 关注度评估完成 "
            f"flow_id={flow_id} @{timestamp:.1f}s operators={operators_turned} "
            f"结果='{key_moment}'"
        )

    def _conclude_attention_if_pending(self) -> None:
        """视频结束时仍开着的关注度窗口用已采样样本兜底收口.

        画面在窗口末端之前就结束（视频过短/采样帧缺失）时，_update_attention 不会再被
        调用；若不兜底，关注度结论将整轮缺失，评估层无法给出"团队关注"维度得分。
        """
        for flow_id in list(self._notice_windows.keys()):
            window = self._notice_windows.get(flow_id)
            if window is None:
                continue
            start = window["start"]
            conclude_sec = min(
                max(self._latest_timestamp, start),
                start + self.NOTICE_ATTENTION_WINDOW_SEC,
            )
            logger.warning(
                f"GazeModule: 视频结束，关注度窗口未自然收口 flow_id={flow_id}，按已采样 "
                f"{len(window['buffer'])} 个工位的注视点兜底收口 @{conclude_sec:.1f}s"
            )
            self._conclude_attention(flow_id, conclude_sec)

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
        # 视频结束仍未收口的关注度窗口先兜底收口，再统一落盘，避免结论缺失
        self._conclude_attention_if_pending()
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
