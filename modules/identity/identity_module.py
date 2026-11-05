"""人员身份与在岗监控模块（独立进程模式）.

按 Hedian 框架契约（BaseModule / BaseStorage / InferenceStream / EventBus）封装。
核心逐帧逻辑在 `IdentityMonitor` 里，与 TrackerModule 内嵌模式共用同一份实现。

模式说明：
    - 独立进程（本文件）：自带检测 + 跟踪 + 身份判定，适合单独跑/调试，代价是多一份 YOLO；
    - 内嵌（config.identity.embedded=true）：由 TrackerModule 复用其 tracks 调用同一个
      IdentityMonitor，零额外检测开销，此时本进程不启动（main.py 控制）。

与其它模块的边界：只依赖注入的 event_bus / config / paths / inference_stream；
不写 Redis Key 字面量、不碰 Web 层、不推视频流（front 通道归 tracker）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from core.base_module import BaseModule
from core.event_bus import EventBus, EventTopic
from core.inference_stream import InferenceStream
from core.path_manager import PathConfig

from modules.identity.identity_monitor import IdentityMonitor
from modules.identity.role_lock import ROLES
from modules.identity.storage_identity import IdentityStorage
from modules.tracker.multi_object_tracker import MultiObjectTracker
from modules.tracker.object_detector import ObjectDetector
from modules.tracker.scene_config import SceneConfig, load_scene

logger = logging.getLogger("module.identity")

# 视频元数据可接受的帧率上限(超过即视为畸形时间戳, 回退配置帧率)
MAX_REASONABLE_FPS = 240.0

_DEFAULT_DET_WEIGHTS = "yolo11_MOT.pt"


class IdentityModule(BaseModule):
    """人员身份与在岗监控模块（独立进程）."""

    def __init__(
        self,
        event_bus: EventBus,
        config: dict,
        paths: PathConfig,
        inference_stream: InferenceStream,
    ) -> None:
        """初始化模块.

        Args:
            event_bus: 跨进程消息总线.
            config: 配置字典（读 identity / tracker / scene 段）.
            paths: 路径配置.
            inference_stream: 推理流写入端.
        """
        super().__init__(event_bus, config, paths, inference_stream)
        self._identity_config: dict = {}
        self._scene: Optional[SceneConfig] = None
        self._detector: Optional[ObjectDetector] = None
        self._tracker: Optional[MultiObjectTracker] = None
        self._monitor: Optional[IdentityMonitor] = None
        self._storage: Optional[IdentityStorage] = None

        self._identity_map: Dict[str, Optional[int]] = {role: None for role in ROLES}
        self._events: List[dict] = []
        self._last_push_ts = -1e9

    @property
    def module_name(self) -> str:
        """模块名称."""
        return "identity"

    # ------------------------------------------------------------------ 初始化
    def initialize(self) -> bool:
        """加载场景、检测器、跟踪器、身份监控组件与存储."""
        try:
            self._identity_config = self.config.get("identity", {}) or {}
            self._scene = load_scene(self.config, self.paths.base_dir)

            detection_config = dict(self.config.get("tracker", {}).get("detection", {}))
            detection_config.update(self._identity_config.get("detection", {}) or {})
            self._detector = ObjectDetector(
                model_path=self.paths.get_model_path(
                    "detection",
                    detection_config.get("weights", _DEFAULT_DET_WEIGHTS),
                ),
                pose_model_path=None,
                conf_threshold=float(detection_config.get("conf_threshold", 0.65)),
                nms_threshold=float(detection_config.get("nms_threshold", 0.35)),
                img_size=int(detection_config.get("img_size", 640)),
            )
            self._tracker = MultiObjectTracker(
                workstations=self._scene.anchors or None
            )

            self._monitor = IdentityMonitor(
                self._scene,
                face_config=self._identity_config.get("face", {}) or {},
                lock_config=self._identity_config.get("lock", {}) or {},
                away_hold_sec=float(self._identity_config.get("away_hold_sec", 2.0)),
            )
            if not self._monitor.initialize():
                return False

            self._storage = IdentityStorage(self.paths)
            logger.info("身份监控模块初始化完成：机室=%s", self._scene.room_id)
            return True
        except Exception as init_error:
            logger.error(f"身份监控模块初始化失败: {init_error}", exc_info=True)
            return False

    # ------------------------------------------------------------------ 主循环
    def process_video(self, video_path: str) -> None:
        """逐帧处理前置视频.

        Args:
            video_path: 视频路径；为空时取 config.videos.front.
        """
        try:
            video_path = self._resolve_video_path(video_path)
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return

            raw_fps = capture.get(cv2.CAP_PROP_FPS)
            if not raw_fps or raw_fps <= 0:
                raise RuntimeError(
                    f"读取视频帧率失败(CAP_PROP_FPS={raw_fps}), 不使用默认帧率"
                )
            fps = float(raw_fps)
            # 畸形时间戳(如 mpegts 经 -c copy 裁剪)会给出 90000 这类帧率
            if fps > MAX_REASONABLE_FPS:
                fallback_fps = float(self.config.get("fps", 30.0))
                logger.warning(
                    f"视频帧率异常(CAP_PROP_FPS={fps}), 回退配置帧率 {fallback_fps}"
                )
                fps = fallback_fps
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            total_sec = total_frames / fps if fps > 0 else 0.0
            push_interval = float(self._identity_config.get("push_interval_sec", 0.5))

            frame_count = 0
            timestamp = 0.0
            logger.info(
                f"开始身份监控: {video_path} (FPS={fps}, 总帧数={total_frames})"
            )

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_count / fps
                frame_count += 1
                self.update_progress(timestamp, total_sec)

                high_det, low_det = self._detector.detect_two_thresholds(frame)
                tracks = self._tracker.track(frame, high_det + low_det)

                locked = self._monitor.process_frame(frame, tracks, timestamp)
                if locked != self._identity_map:
                    self._on_role_change(timestamp, locked)

                for record in self._monitor.drain_events():
                    self._publish_identity_event(record)

                if timestamp - self._last_push_ts >= push_interval:
                    self._last_push_ts = timestamp
                    self._push_state(timestamp)

            self._monitor.close(timestamp)
            for record in self._monitor.drain_events():
                self._publish_identity_event(record)

            logger.info(
                "身份监控处理完成：帧数=%d 时长=%.1fs 事件=%d 离岗段=%d",
                frame_count,
                timestamp,
                len(self._monitor.events),
                len(self._monitor.away_events),
            )
        except Exception as exc:
            logger.error(f"身份监控处理失败: {exc}", exc_info=True)
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ------------------------------------------------------------------ 结果落盘
    def save_results(self, run_id: str) -> None:
        """保存 key_moments / 离岗明细 / 角色快照.

        Args:
            run_id: 本次运行标识.
        """
        if self._storage is None or self._monitor is None:
            logger.warning("存储或监控未初始化，跳过保存")
            return
        self._storage.save_key_moments(run_id, self._monitor.events)
        self._storage.save_away_events(run_id, self._monitor.away_events)
        self._storage.save_role_info(
            run_id, self._monitor.snapshot, self._scene.room_id if self._scene else ""
        )
        logger.info(
            "身份监控结果已保存：%d 事件 / %d 离岗段",
            len(self._monitor.events),
            len(self._monitor.away_events),
        )

    # ------------------------------------------------------------------ 内部工具
    def _resolve_video_path(self, video_path: str) -> str:
        """解析视频路径（相对路径按项目根，缺省取 config.videos.front）."""
        if not video_path:
            video_path = str(
                self.paths.base_dir
                / self.config.get("videos", {}).get("front", "data/videos/camFRONT.mpg")
            )
        elif not Path(video_path).is_absolute() and not Path(video_path).exists():
            video_path = str(self.paths.base_dir / video_path)
        return video_path

    def _on_role_change(
        self, timestamp: float, locked: Dict[str, Optional[int]]
    ) -> None:
        """角色锁定结果变化：记录关键事件并广播."""
        newly = {
            role: tid
            for role, tid in locked.items()
            if tid is not None and self._identity_map.get(role) != tid
        }
        self._identity_map = locked
        if not newly:
            return

        roles_text = ",".join(
            f"{role}:{tid}" for role, tid in locked.items() if tid is not None
        )
        record = {
            "localSec": round(timestamp, 2),
            "key_moment": f"角色分配,{roles_text}",
            "source": "identity",
            "identities": {r: t for r, t in locked.items() if t is not None},
        }
        self._monitor.events.append(record)
        self.push_display(
            "identity",
            {
                "localSec": round(timestamp, 2),
                "tag": "ROLE_ASSIGNED",
                "data": {"roles": record["identities"]},
            },
        )
        self.push_event(
            EventTopic.IDENTITY_ROLE,
            {
                "localSec": round(timestamp, 2),
                "roles": record["identities"],
                "room_id": self._scene.room_id if self._scene else "",
            },
            timestamp=timestamp,
        )
        logger.info(f"角色锁定更新: {record['identities']}")

    def _publish_identity_event(self, record: dict) -> None:
        """把离岗事件发到事件总线（前端时间轴/规则层消费）."""
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

    def _push_state(self, timestamp: float) -> None:
        """推送各身份在岗状态到前端面板."""
        snapshot = self._monitor.snapshot if self._monitor else {}
        self.push_display(
            "identity",
            {
                "localSec": round(timestamp, 2),
                "tag": "IDENTITY_STATE",
                "data": {
                    "room_id": self._scene.room_id if self._scene else "",
                    "identities": snapshot,
                    "templates": self._monitor.template_counts() if self._monitor else {},
                },
            },
        )
        self.inference_stream.update_module_snapshot(
            "identity",
            {
                "roles": dict(self._identity_map),
                "on_station": {
                    name: info.get("on_station") for name, info in snapshot.items()
                },
                "room_id": self._scene.room_id if self._scene else "",
                "localSec": round(timestamp, 2),
            },
        )
