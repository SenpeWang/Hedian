"""
凝视处理器 — 独立模块，MOT调用

负责：头部检测、注视推断、ROI分类、可视化、推送推理结果
"""
import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any, Callable, List

from modules.gaze.head_detector import HeadDetector
from modules.gaze.gaze_estimator import GazeEstimator
from modules.gaze.roi_classifier import ROIClassifier
from modules.gaze.gaze_attention import GazePoint, GazeAttentionChecker
from modules.gaze.storage_gaze import GazeStorage

logger = logging.getLogger("module.gaze.processor")

# 状态颜色（BGR）
COLOR_MAP = {
    "IN_ROI": (0, 255, 0),       # 绿色
    "OUTSIDE_ROI": (0, 0, 255),   # 红色
    "OUTSIDE_FRAME": (0, 255, 255),  # 黄色
}


class GazeModule:
    """
    凝视处理器

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
        display_fn: Callable = None,
        event_bus=None,
        progress_fn: Callable = None,
        paths=None,
    ):
        """
        初始化凝视处理器

        Args:
            head_model_path: 头部检测模型路径
            gaze_model_path: 注视推断模型路径
            roi_json_path: ROI配置文件路径
            config: 凝视配置
            display_fn: 推送推理结果的函数 (event_type, data) -> None
            event_bus: 消息总线（用于关键事件通信）
            progress_fn: 更新进度的函数 (current, total) -> None
            paths: PathConfig 实例，用于结果保存
        """
        config = config or {}

        self._event_bus = event_bus
        self._progress_fn = progress_fn
        self._storage = GazeStorage(paths) if paths is not None else None

        # 信息通报流程激活标志
        self._info_notice_active = False

        # 订阅流程开始/结束事件
        if self._event_bus is not None:
            try:
                from core.event_bus import EventTopic
                self._event_bus.subscribe(EventTopic.FLOW_STARTED, self._on_flow_started)
                self._event_bus.subscribe(EventTopic.FLOW_ENDED, self._on_flow_ended)
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

        self._display_fn = display_fn

        # 缓存
        self._cached_results = []
        self._cached_has_heads = False
        self._cached_any_in_roi = False
        self._gaze_interval = 10

        # 告警状态
        self._away_start_ts = None
        self._alerting = False
        self._away_threshold = 60.0

        # 事件记录
        self._events = []
        self._latest_ts = 0.0

        # 注视转动判定
        self._attention_checker = GazeAttentionChecker(
            min_turn_displacement=config.get("min_turn_displacement", 100.0),
            min_samples=config.get("min_gaze_samples", 5),
        )
        self._attn_window_interval = config.get("attention_window_interval", 3.0)
        self._attn_label_duration = 2.0
        self._attn_window_start = 0.0
        self._attn_buffer: List[GazePoint] = []
        self._attn_label = None

        logger.info("凝视处理器初始化完成")

    def process_frame(self, frame: np.ndarray, ts: float, frame_count: int) -> np.ndarray:
        """处理一帧：检测、估计、可视化、推送"""
        self._latest_ts = ts
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 画 ROI 区域
        self._draw_rois(vis)
        # 画 head_zones
        self._draw_head_zones(vis)

        # 每N帧做一次凝视检测
        if frame_count % self._gaze_interval == 0:
            self._run_gaze_detection(frame, w, h, ts)

        # 注视转动判定
        self._update_attention(ts)

        # 更新进度时间
        if self._progress_fn:
            self._progress_fn(ts, None)

        # 画头部框+注视线+注视点
        self._draw_gaze_results(vis, ts)

        return vis

    def _run_gaze_detection(self, frame: np.ndarray, w: int, h: int, ts: float):
        """运行凝视检测"""
        all_heads = self._head_detector.detect(frame)
        # 过滤：头部在 head_zones 内
        heads = self._roi_classifier.filter_heads_by_zone(all_heads)

        self._cached_results = []
        self._cached_has_heads = bool(heads)
        self._cached_any_in_roi = False

        if heads:
            # 凝视推断
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
                    self._cached_results.append({
                        "box": (box.x1, box.y1, box.x2, box.y2),
                        "center": (box.cx, box.cy),
                        "gaze_pt": gaze_pt,
                        "status": status,
                    })

        # 计算当前 away 时长（无人注视盘台的持续秒数，用于前端进度条展示）
        away_dur = 0.0
        if self._cached_has_heads and not self._cached_any_in_roi and self._away_start_ts is not None:
            away_dur = ts - self._away_start_ts

        # 推送凝视推理结果到推理流
        if self._display_fn:
            self._display_fn("gaze", {
                "localSec": round(ts, 2),
                "tag": "gaze_status",
                "data": {
                    "has_heads": self._cached_has_heads,
                    "any_in_roi": self._cached_any_in_roi,
                    "heads_count": len(self._cached_results),
                    "away_duration": round(away_dur, 2),
                },
            })

        # 告警逻辑：无人注视超过60秒
        self._check_alert(ts)

    def _check_alert(self, ts: float):
        """检查告警条件"""
        if self._cached_has_heads and not self._cached_any_in_roi:
            if self._away_start_ts is None:
                self._away_start_ts = ts
            away_dur = ts - self._away_start_ts
            if away_dur >= self._away_threshold and not self._alerting:
                self._alerting = True
                # 推理流：data 只含纯展示字段
                if self._display_fn:
                    self._display_fn("gaze", {
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
                    self._event_bus.publish(EventTopic.GAZE_ALERT, {
                        "localSec": round(ts, 2),
                        "state": "无人注视盘台",
                        "away_duration": round(away_dur, 2),
                        "heads_count": len(self._cached_results),
                    }, ts=ts)
                logger.warning(f"凝视告警: 无人注视盘台 {away_dur:.1f}秒 @{ts:.1f}s")
        else:
            if self._alerting:
                # 违规结束，记录持续秒数到关键时刻
                duration = ts - self._away_start_ts if self._away_start_ts else 0.0
                self._events.append({
                    "localSec": round(self._away_start_ts, 2),
                    "key_moment": f"没有看盘台持续{round(duration, 1)}秒",
                })
                # 推理流：通知前端告警结束
                if self._display_fn:
                    self._display_fn("gaze", {
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
                    }, ts=ts)
            self._away_start_ts = None
            self._alerting = False

    def _on_flow_started(self, msg: dict) -> None:
        """流程开始事件回调：信息通报流程开始时，激活 ATTENTION_RESULT 推送"""
        data = msg.get("data", {})
        if data.get("flow_type") == "info_notice":
            self._info_notice_active = True
            logger.info("GazeModule: 信息通报流程激活")

    def _on_flow_ended(self, msg: dict) -> None:
        """流程结束事件回调：信息通报流程结束时，关闭 ATTENTION_RESULT 推送"""
        data = msg.get("data", {})
        if data.get("flow_type") == "info_notice":
            self._info_notice_active = False
            logger.info("GazeModule: 信息通报流程结束")

    def _update_attention(self, ts: float):
        """收集当前帧注视点到窗口缓冲区，窗口结束时评估

        推送策略：推理流 ATTENTION_RESULT 仅在信息通报流程激活时推送；
        事件流 GAZE_ATTENTION 始终发布，供 info_notice_rule 规则订阅判定。
        """
        # 从当前帧缓存结果中取所有注视点的均值作为代表点
        if self._cached_results:
            gx_mean = sum(r["gaze_pt"][0] for r in self._cached_results) / len(self._cached_results)
            gy_mean = sum(r["gaze_pt"][1] for r in self._cached_results) / len(self._cached_results)
            self._attn_buffer.append(GazePoint(ts * 1000.0, gx_mean, gy_mean))

        # 检查窗口是否结束
        if ts >= self._attn_window_start + self._attn_window_interval:
            if self._attn_buffer:
                result = self._attention_checker.evaluate(self._attn_buffer)
                self._attn_label = (result.has_turned, ts + self._attn_label_duration)
                # 推理流：仅在信息通报流程激活时推送
                if self._display_fn and self._info_notice_active:
                    self._display_fn("gaze", {
                        "localSec": round(ts, 2),
                        "tag": "ATTENTION_RESULT",
                        "data": {
                            "has_turned": result.has_turned,
                            "displacement": round(result.total_displacement, 1),
                        },
                    })
                # 未转动则记录至关键时刻
                if not result.has_turned and self._info_notice_active:
                    self._events.append({
                        "localSec": round(self._attn_window_start, 2),
                        "key_moment": "没有给予关注",
                    })
                # 事件流：始终发布
                if self._event_bus:
                    from core.event_bus import EventTopic
                    self._event_bus.publish(EventTopic.GAZE_ATTENTION, {
                        "localSec": round(ts, 2),
                        "has_turned": result.has_turned,
                        "displacement": round(result.total_displacement, 1),
                        "sample_count": result.sample_count,
                        "reason": result.reason,
                        "window": f"{self._attn_window_start:.1f}s~{ts:.1f}s",
                    }, ts=ts)
                logger.info(
                    "注视转动窗口 %.1f-%.1fs: %s (位移 %.1f px, %d 样本) info_notice_active=%s",
                    self._attn_window_start, ts,
                    "已关注" if result.has_turned else "未关注",
                    result.total_displacement, result.sample_count,
                    self._info_notice_active,
                )
            # 开启下一个窗口
            self._attn_window_start = ts
            self._attn_buffer = []

    def _draw_rois(self, vis: np.ndarray):
        """画ROI区域（半透明黄色）"""
        rois = self._roi_classifier._gaze_rois
        if not rois:
            return
        overlay = vis.copy()
        for label, contour in rois:
            pts = contour.astype(np.int32).reshape(-1, 2)
            cv2.fillPoly(overlay, [pts], (0, 200, 255))
            cv2.polylines(vis, [pts], True, (0, 200, 255), 2)
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(vis, label, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

    def _draw_head_zones(self, vis: np.ndarray):
        """画head_zones（绿色虚线）"""
        zones = self._roi_classifier._head_zones
        if not zones:
            return
        for label, contour in zones:
            pts = contour.astype(np.int32).reshape(-1, 2)
            for j in range(len(pts)):
                if j % 2 == 0:
                    cv2.line(vis, tuple(pts[j]), tuple(pts[(j+1)%len(pts)]), (0, 255, 0), 1, cv2.LINE_AA)

    def _draw_gaze_results(self, vis: np.ndarray, ts: float = 0.0):
        """画头部框+注视线+注视点+关注度标签"""
        for gr in self._cached_results:
            color = COLOR_MAP.get(gr["status"], (255, 255, 255))
            x1, y1, x2, y2 = map(int, gr["box"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            gx, gy = gr["gaze_pt"]
            cx, cy = gr["center"]
            cv2.line(vis, (cx, cy), (gx, gy), color, 2, cv2.LINE_AA)
            cv2.circle(vis, (gx, gy), 4, color, -1)
            cv2.circle(vis, (gx, gy), 6, color, 2)

        # 绘制关注度判定标签（窗口评估后显示 2 秒）
        if self._attn_label is not None:
            has_turned, expire_ts = self._attn_label
            if ts <= expire_ts:
                label = "Attended" if has_turned else "Not Attended"
                label_color = (0, 200, 0) if has_turned else (0, 0, 220)
                h_vis = vis.shape[0]
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                tx = (vis.shape[1] - text_size[0]) // 2
                ty = h_vis - 20
                overlay = vis.copy()
                cv2.rectangle(overlay, (tx - 10, ty - text_size[1] - 10),
                              (tx + text_size[0] + 10, ty + 10), label_color, -1)
                cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
                cv2.putText(vis, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                self._attn_label = None

    def get_events(self) -> list:
        """获取凝视关键时刻事件"""
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
        """保存凝视关键事件到 gaze/gaze_key_moments.json（委托给 GazeStorage）"""
        self._storage.save_key_moments(run_id, self.get_events())
