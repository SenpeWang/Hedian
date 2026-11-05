"""身份监控核心组件（不绑视频读取，供内嵌与独立进程共用）.

设计目的：把"逐帧身份判定 + 在岗/离岗状态机"从视频循环里剥出来，
使得同一份逻辑既能被独立的 IdentityModule 驱动，也能被 TrackerModule 内嵌复用
（内嵌时直接吃 tracker 已有的 tracks，避免重复跑 YOLO）。

调用方职责：提供 frame / tracks / timestamp；消费 `identity_map` 与 `drain_events()`。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from modules.identity.face_recognizer import FaceRecognizer
from modules.identity.role_lock import ROLES, RoleLock
from modules.identity.station_state import StationMonitor
from modules.tracker.scene_config import SceneConfig

logger = logging.getLogger("module.identity.monitor")

_IDENTITY_CN = {"LEADER": "机组长", "ROAD1": "操作员1", "ROAD2": "操作员2"}


class IdentityMonitor:
    """人脸身份锁定 + 在岗/离岗判定（逐帧组件）."""

    def __init__(
        self,
        scene: SceneConfig,
        face_config: Optional[dict] = None,
        lock_config: Optional[dict] = None,
        away_hold_sec: float = 2.0,
    ) -> None:
        """初始化.

        Args:
            scene: 当前机室场景（工位多边形 / 机室号）.
            face_config: identity.face 配置段.
            lock_config: identity.lock 配置段.
            away_hold_sec: 离岗防抖秒数.
        """
        self._scene = scene
        self._face_config = face_config or {}
        self._lock_config = lock_config or {}
        self._away_hold = float(away_hold_sec)

        self._face: Optional[FaceRecognizer] = None
        self._lock: Optional[RoleLock] = None
        self._station: Optional[StationMonitor] = None

        self._emb_cache: Dict[int, Tuple[float, Optional[np.ndarray]]] = {}
        self._verify_interval = float(self._face_config.get("verify_interval_sec", 0.5))

        self.identity_map: Dict[str, Optional[int]] = {role: None for role in ROLES}
        self.events: List[dict] = []
        self.away_events: List[dict] = []
        self.snapshot: Dict[str, dict] = {}
        self._queue: List[dict] = []

    # ------------------------------------------------------------------ 初始化
    def initialize(self) -> bool:
        """构建人脸识别器、锁定器与在岗状态机.

        Returns:
            初始化成功返回 True.
        """
        try:
            polygons_px: Dict[str, list] = {}
            for role in ROLES:
                pts = self._scene.polygon_px(role)
                if pts:
                    polygons_px[role] = pts
            if not polygons_px:
                logger.warning(
                    "机室 %s 未配置工位多边形，在岗判定不生效", self._scene.room_id
                )

            if self._face_config.get("enabled", True):
                self._face = FaceRecognizer(
                    model_name=str(self._face_config.get("model_name", "buffalo_l")),
                    det_size=int(self._face_config.get("det_size", 640)),
                    ctx_id=int(self._face_config.get("ctx_id", 0)),
                    min_box_size=int(self._face_config.get("min_box_size", 40)),
                    det_score=float(self._face_config.get("det_score", 0.7)),
                )
            else:
                logger.warning("人脸识别已关闭，身份将只依赖排除法补位")

            cfg = self._lock_config
            self._lock = RoleLock(
                polygons_px,
                spatial_threshold=float(cfg.get("spatial_threshold", 1.2)),
                margin=float(cfg.get("margin", 0.10)),
                grace_sec=float(cfg.get("grace_sec", 2.5)),
                strong_time_sec=float(cfg.get("strong_time_sec", 1.5)),
                strong_passes=int(cfg.get("strong_passes", 2)),
                weak_time_sec=float(cfg.get("weak_time_sec", 5.0)),
                weak_passes=int(cfg.get("weak_passes", 4)),
                absent_hold_sec=float(cfg.get("absent_hold_sec", 3.0)),
                elim_frames=int(cfg.get("elim_frames", 5)),
                auto_register_frames=int(cfg.get("auto_register_frames", 15)),
            )
            self._station = StationMonitor(polygons_px, away_hold_sec=self._away_hold)
            logger.info(
                "身份监控组件就绪：机室=%s 工位多边形=%d 人脸=%s",
                self._scene.room_id, len(polygons_px), "开" if self._face else "关",
            )
            return True
        except Exception as exc:
            logger.error("身份监控组件初始化失败: %s", exc, exc_info=True)
            return False

    @property
    def has_templates(self) -> bool:
        """是否已建立任一角色底库."""
        return bool(self._lock and any(self._lock.has_templates(r) for r in ROLES))

    def template_counts(self) -> Dict[str, int]:
        """返回各角色底库模板条数（用于面板/日志）."""
        if self._lock is None:
            return {role: 0 for role in ROLES}
        return {role: len(self._lock.templates.get(role) or []) for role in ROLES}

    # ------------------------------------------------------------------ 主入口
    def process_frame(
        self, frame: np.ndarray, tracks, timestamp: float
    ) -> Dict[str, Optional[int]]:
        """处理单帧.

        Args:
            frame: BGR 原始帧.
            tracks: 当前帧 STrack 列表（须含 track_id / bbox / get_center）.
            timestamp: 视频时间（秒）.

        Returns:
            {角色: 已锁定 tid 或 None}
        """
        if self._lock is None or self._station is None:
            return self.identity_map

        people = self._build_people(frame, tracks, timestamp)
        locked = self._lock.update(timestamp, people, {p["tid"] for p in people})
        self.identity_map = locked

        centers = self._centers(tracks)
        active = {role for role, tid in locked.items() if tid is not None}
        for event in self._station.update(timestamp, centers, active=active):
            self._record_event(event)
        self.snapshot = self._build_snapshot(timestamp, centers)
        return locked

    def close(self, timestamp: float) -> None:
        """视频结束时闭合未结束的离岗段.

        Args:
            timestamp: 末帧时间（秒）.
        """
        if self._station is None:
            return
        for event in self._station.close(timestamp):
            self._record_event(event)

    def drain_events(self) -> List[dict]:
        """取走本帧累积的新事件.

        Returns:
            事件列表，元素含 timestamp / key_moment / identity / event / duration_sec.
        """
        queued, self._queue = self._queue, []
        return queued

    # ------------------------------------------------------------------ 内部
    def _build_people(self, frame, tracks, timestamp: float) -> List[dict]:
        """整理身份锁定输入，按节流缓存人脸特征."""
        people: List[dict] = []
        alive: Set[int] = set()
        locked_tids = {int(t) for t in self.identity_map.values() if t is not None}
        need_verify_all = any(t is None for t in self.identity_map.values())

        for track in tracks:
            tid = int(track.track_id)
            alive.add(tid)
            box = [float(v) for v in track.bbox[:4]]
            cx, cy = track.get_center()
            emb = None
            if self._face is not None and (need_verify_all or tid not in locked_tids):
                cached = self._emb_cache.get(tid)
                if cached is not None and timestamp - cached[0] < self._verify_interval:
                    emb = cached[1]
                else:
                    emb = self._face.extract(frame, box)
                    self._emb_cache[tid] = (timestamp, emb)
            people.append(
                {"tid": tid, "box": box, "center": (float(cx), float(cy)), "emb": emb}
            )
        for tid in list(self._emb_cache.keys()):
            if tid not in alive:
                self._emb_cache.pop(tid, None)
        return people

    def _centers(self, tracks) -> Dict[str, Optional[tuple]]:
        """按已锁定身份取角色对应的人体中心点."""
        centers: Dict[str, Optional[tuple]] = {role: None for role in ROLES}
        for track in tracks:
            tid = int(track.track_id)
            for role, locked_tid in self.identity_map.items():
                if locked_tid is not None and int(locked_tid) == tid:
                    cx, cy = track.get_center()
                    centers[role] = (float(cx), float(cy))
        return centers

    def _build_snapshot(
        self, timestamp: float, centers: Dict[str, Optional[tuple]]
    ) -> Dict[str, dict]:
        """生成前端面板所需的状态快照."""
        snapshot: Dict[str, dict] = {}
        for role in ROLES:
            center = centers.get(role)
            on_station = (
                self._station.is_on_station(role, center) if self._station else False
            )
            away_sec = (
                self._station.away_seconds(role, timestamp) if self._station else 0.0
            )
            snapshot[role] = {
                "tid": self.identity_map.get(role),
                "on_station": on_station,
                "away": bool(self._station and self._station.is_away(role)),
                "away_sec": round(away_sec, 1),
                "center": [int(center[0]), int(center[1])] if center else None,
            }
        return snapshot

    def _record_event(self, event: dict) -> None:
        """累积离岗事件，并生成 key_moment 文案入队."""
        identity = event.get("identity", "")
        cn = _IDENTITY_CN.get(identity, identity)
        if event["type"] == "away_start":
            self.away_events.append(
                {
                    "identity": identity,
                    "start_sec": round(float(event["start"]), 2),
                    "end_sec": None,
                    "duration_sec": None,
                }
            )
            key_moment = f"{cn}离岗"
            moment_ts = float(event["start"])
        else:
            for item in self.away_events:
                if item["identity"] == identity and item["end_sec"] is None:
                    item["end_sec"] = round(float(event["end"]), 2)
                    item["duration_sec"] = event.get("duration")
                    break
            key_moment = (
                f"{cn}返岗（离岗 {event.get('duration')}s）"
                if event.get("duration") is not None
                else f"{cn}返岗"
            )
            moment_ts = float(event["end"] or event["start"])

        record = {
            "localSec": round(moment_ts, 2),
            "key_moment": key_moment,
            "source": "identity",
            "identity": identity,
            "event": event["type"],
            "duration_sec": event.get("duration"),
        }
        self.events.append(record)
        self._queue.append(record)
        logger.info("身份事件 %.1fs %s", moment_ts, key_moment)
