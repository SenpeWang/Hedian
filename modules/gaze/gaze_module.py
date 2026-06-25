"""
注视检测模块入口

逻辑：检测操作员是否看向盘台(ROI)
- 有人看盘台 → 正常，不推送不保存
- 没人看盘台（但检测到人）→ 告警，推送并保存
"""
import os
import json
import logging

from core.base_module import BaseModule
from core.message_bus import MessageBus, MsgType
from core.frontend_sync import FrontendSync
from core.path_manager import PathConfig

from modules.gaze.head_detector import HeadDetector
from modules.gaze.gaze_estimator import GazeEstimator
from modules.gaze.roi_classifier import ROIClassifier

logger = logging.getLogger("module.gaze")


class GazeModule(BaseModule):
    """
    注视检测模块

    负责头部检测、注视推断和 ROI 分类。
    只在检测到人但无人注视盘台时产生告警事件。
    """

    def __init__(
        self,
        bus: MessageBus,
        config: dict,
        paths: PathConfig,
        aggregator: FrontendSync,
    ):
        super().__init__(bus, config, paths, aggregator)
        self._head_detector = None
        self._gaze_estimator = None
        self._roi_classifier = None
        self._events = []  # 只保存告警事件

    @property
    def module_name(self) -> str:
        return "gaze"

    def initialize(self) -> bool:
        """初始化注视检测模块"""
        try:
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

            logger.info("注视检测模块初始化完成")
            return True

        except Exception as e:
            logger.error(f"注视检测模块初始化失败: {e}", exc_info=True)
            return False

    def process_video(self, video_path: str) -> None:
        """处理视频，进行注视检测"""
        import cv2

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or self.config.get("fps", 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            # 连续无人注视计数器（防抖：连续N帧无人注视才告警）
            no_gaze_count = 0
            NO_GAZE_THRESHOLD = 10  # 连续10帧（约0.33秒）无人注视才告警

            # 上一次告警时间（避免重复告警）
            last_alert_ts = -999.0
            ALERT_COOLDOWN = 5.0  # 两次告警间隔至少5秒

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

                # 头部检测
                heads = self._head_detector.detect(frame)
                heads = self._roi_classifier.filter_heads_by_zone(heads)

                if not heads:
                    # 没检测到人，不告警（可能画面中没人）
                    no_gaze_count = 0
                    continue

                # 注视推断
                heatmaps, inout_scores, valid_boxes = self._gaze_estimator.predict(frame, heads)

                if heatmaps is None or not valid_boxes:
                    continue

                # 检查是否有人注视 ROI
                any_in_roi = False
                for i, box in enumerate(valid_boxes):
                    heatmap = heatmaps[i]
                    if heatmap.ndim == 3:
                        heatmap = heatmap[0]

                    inout_score = float(inout_scores[i]) if inout_scores is not None else 1.0
                    gaze_pt = self._roi_classifier.extract_gaze_point(heatmap, frame.shape[1], frame.shape[0])

                    if gaze_pt is None:
                        continue

                    status, roi_label = self._roi_classifier.classify_gaze(inout_score, gaze_pt)

                    if status == "IN_ROI":
                        any_in_roi = True
                        break  # 有人看盘台就够了，不用继续

                # 告警逻辑：有人但没人看盘台
                if any_in_roi:
                    no_gaze_count = 0
                else:
                    no_gaze_count += 1

                    # 连续无人注视达到阈值 + 冷却时间已过
                    if (no_gaze_count >= NO_GAZE_THRESHOLD
                            and (ts - last_alert_ts) >= ALERT_COOLDOWN):

                        ev = {
                            "localSec": round(ts, 2),
                            "source": "gaze",
                            "event": "GAZE_ALERT",
                            "state": "无人注视盘台",
                            "heads_count": len(heads),
                            "frame_id": frame_count,
                        }
                        self._events.append(ev)

                        # 推送到聚合器
                        self.push_event("gaze", ev)

                        # 发布到总线
                        self.bus.publish(MsgType.GAZE_ALERT, {
                            "localSec": round(ts, 2),
                            "heads_count": len(heads),
                        }, ts=ts)

                        last_alert_ts = ts
                        no_gaze_count = 0  # 重置，等待下一次告警

                # 进度日志
                if frame_count % 300 == 0:
                    pct = frame_count * 100 // total_frames if total_frames else 0
                    roi_info = " | 盘台:" + ("✅" if any_in_roi else "❌") if heads else ""
                    logger.info(f"{frame_count}/{total_frames}帧 {pct}%{roi_info}")

            cap.release()
            logger.info(f"视频处理完成，共 {frame_count} 帧，告警 {len(self._events)} 次")

        except Exception as e:
            logger.error(f"视频处理失败: {e}", exc_info=True)

    def save_results(self, run_id: str) -> None:
        """保存注视检测结果（只保存告警事件）"""
        if not self._events:
            logger.info("没有告警事件，跳过保存")
            return

        # 保存告警事件
        output_path = self.paths.get_result_path(
            run_id=run_id,
            module="gaze",
            filename="Gaze_alerts.json",
        )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self._events, f, ensure_ascii=False, indent=2)
            logger.info(f"保存 {len(self._events)} 个告警事件到 {output_path}")
        except Exception as e:
            logger.error(f"保存告警事件失败: {e}", exc_info=True)
