"""三角色身份锁定状态机（人脸底库 + 驻留确认 + 排除法补位）.

从 Nuclear_autoface_enhanced 的 identity 逻辑移植，保留两条关键修复：
    1. 驻留确认用「判定区内有效累积时长 acc」，不用墙钟，避免"人已走开 stay 照样涨"的假通过；
    2. 取消策略统一走 grace 超时，容忍 ID 抖动，与继承逻辑一致。

角色约定与 Hedian 对齐：LEADER / ROAD1（右操作盘）/ ROAD2（左操作盘）。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.identity.face_recognizer import face_distance
from modules.identity.station_state import point_in_polygon

logger = logging.getLogger("module.identity.lock")

ROLES: Tuple[str, ...] = ("LEADER", "ROAD1", "ROAD2")
Polygon = Sequence[Tuple[float, float]]


class RoleLock:
    """三角色身份锁定器.

    输入每帧的人体跟踪结果（tid / 人体框 / 中心 / 可选人脸特征），输出
    `{角色: 已锁定 tid}`。未锁定的角色值为 None。
    """

    def __init__(
        self,
        polygons_px: Dict[str, Polygon],
        spatial_threshold: float = 1.2,
        margin: float = 0.10,
        aggregation: str = "topk_mean",
        topk_k: int = 3,
        grace_sec: float = 2.5,
        strong_time_sec: float = 1.5,
        strong_passes: int = 2,
        weak_time_sec: float = 5.0,
        weak_passes: int = 4,
        absent_hold_sec: float = 3.0,
        elim_frames: int = 5,
        auto_register_frames: int = 15,
        max_templates_per_role: int = 40,
    ) -> None:
        """初始化.

        Args:
            polygons_px: 角色 -> 工位像素多边形（排除法补位用）.
            spatial_threshold: 空间验证阈值，距离小于此值判为匹配.
            margin: 竞争余量，最佳角色需比次优好出该值以上.
            aggregation: 距离聚合方式 min / topk_mean.
            topk_k: topk_mean 的 K.
            grace_sec: 驻留确认允许的中断间隔.
            strong_time_sec: 强匹配（距离很近）所需驻留时长.
            strong_passes: 强匹配所需合格采样次数.
            weak_time_sec: 弱匹配所需驻留时长.
            weak_passes: 弱匹配所需采样次数.
            absent_hold_sec: 锁定后连续不在画面多久释放该角色.
            elim_frames: 排除法补位需要的连续帧数.
            auto_register_frames: 自动注册底库需要的连续在岗帧数.
            max_templates_per_role: 每角色底库模板上限（FIFO 淘汰）.
        """
        self._polygons = dict(polygons_px or {})
        self._spatial_threshold = float(spatial_threshold)
        self._margin = float(margin)
        self._agg = aggregation
        self._topk = int(topk_k)
        self._grace = float(grace_sec)
        self._strong_time = float(strong_time_sec)
        self._strong_passes = int(strong_passes)
        self._weak_time = float(weak_time_sec)
        self._weak_passes = int(weak_passes)
        self._absent_hold = float(absent_hold_sec)
        self._elim_frames = int(elim_frames)
        self._auto_reg_frames = int(auto_register_frames)
        self._max_templates = int(max_templates_per_role)

        self.templates: Dict[str, List[np.ndarray]] = {r: [] for r in ROLES}
        self.locked: Dict[str, Optional[int]] = {r: None for r in ROLES}
        # role -> {"tid": int, "acc": float, "passes": int, "last": float}
        self.pending: Dict[str, dict] = {}
        self._absent_since: Dict[int, float] = {}
        # role -> {"tid": int, "frames": int}（自动注册/排除法共用的连续帧计数）
        self._station_streak: Dict[str, dict] = {}
        self._elim_streak: Dict[str, dict] = {}

    # ------------------------------------------------------------------ 查询
    def has_templates(self, role: str) -> bool:
        """该角色是否已有底库模板."""
        return bool(self.templates.get(role))

    def template_matrix(self, role: str) -> Optional[np.ndarray]:
        """取该角色底库矩阵 (N, D)，无模板返回 None."""
        tpl = self.templates.get(role) or []
        return np.vstack(tpl) if tpl else None

    def register(
        self, role: str, embedding: np.ndarray, cross_exclusive: float = 0.80
    ) -> bool:
        """向指定角色底库追加一条模板.

        Args:
            role: 角色名.
            embedding: 已归一化的人脸特征.
            cross_exclusive: 与其它角色底库距离小于该值时拒绝注册（防串人）.

        Returns:
            是否注册成功.
        """
        if role not in self.templates or embedding is None:
            return False
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        for other in ROLES:
            if other == role or not self.templates.get(other):
                continue
            if face_distance(emb, self.template_matrix(other), self._agg, self._topk) < cross_exclusive:
                logger.warning("拒绝对 %s 注册：与 %s 底库过近", role, other)
                return False
        bucket = self.templates[role]
        bucket.append(emb)
        if len(bucket) > self._max_templates:
            bucket.pop(0)
        return True

    # ------------------------------------------------------------------ 主更新
    def update(
        self,
        timestamp: float,
        people: List[dict],
        present_tids: Optional[set] = None,
    ) -> Dict[str, Optional[int]]:
        """用本帧观测更新锁定状态.

        Args:
            timestamp: 视频时间（秒）.
            people: 列表，元素含 tid / box / center / emb(可选).
            present_tids: 画面内的 tid 集合；None 时由 people 推出.

        Returns:
            {角色: tid 或 None}
        """
        if present_tids is None:
            present_tids = {int(p["tid"]) for p in people}

        self._release_absent(timestamp, present_tids)
        self._auto_register(timestamp, people)
        self._confirm_by_face(timestamp, people)
        self._assign_by_elimination(timestamp, people, present_tids)
        self._tick_pending(timestamp, present_tids)
        return dict(self.locked)

    # ------------------------------------------------------------------ 内部
    def _release_absent(self, timestamp: float, present_tids: set) -> None:
        """锁定后连续不在画面超过阈值 → 释放角色."""
        for role, tid in self.locked.items():
            if tid is None:
                continue
            if tid in present_tids:
                self._absent_since.pop(tid, None)
                continue
            since = self._absent_since.setdefault(tid, timestamp)
            if timestamp - since >= self._absent_hold:
                logger.info(
                    "释放角色 %s（tid=%s 已 %.1fs 不在画面）", role, tid, timestamp - since
                )
                self.locked[role] = None
                self.pending.pop(role, None)
                self._absent_since.pop(tid, None)

    def _auto_register(self, timestamp: float, people: List[dict]) -> None:
        """某人在角色工位稳定停留 → 自动把其人脸注册为该角色底库."""
        for role in ROLES:
            if self.templates.get(role):
                continue
            polygon = self._polygons.get(role)
            if not polygon:
                continue
            occupant = None
            for person in people:
                center = person.get("center")
                if center and point_in_polygon(center, polygon):
                    occupant = person
                    break
            streak = self._station_streak.get(role) or {"tid": None, "frames": 0}
            if occupant is None:
                self._station_streak[role] = {"tid": None, "frames": 0}
                continue
            if streak.get("tid") != occupant["tid"]:
                streak = {"tid": occupant["tid"], "frames": 0}
            streak["frames"] = int(streak["frames"]) + 1
            self._station_streak[role] = streak
            emb = occupant.get("emb")
            if streak["frames"] >= self._auto_reg_frames and emb is not None:
                if self.register(role, emb):
                    logger.info(
                        "自动注册底库 %s ← tid=%s（连续在岗 %d 帧）",
                        role, occupant["tid"], streak["frames"],
                    )
                    self._station_streak[role] = {"tid": None, "frames": 0}

    def _confirm_by_face(self, timestamp: float, people: List[dict]) -> None:
        """人脸匹配 + 驻留确认 → 锁定角色."""
        for role in ROLES:
            if self.locked.get(role) is not None:
                continue
            templates = self.template_matrix(role)
            if templates is None:
                continue
            best: Optional[tuple] = None  # (tid, emb, dist)
            for person in people:
                if person.get("tid") in self.locked.values():
                    continue
                emb = person.get("emb")
                if emb is None:
                    continue
                dist = face_distance(emb, templates, self._agg, self._topk)
                if dist >= self._spatial_threshold:
                    continue
                if best is None or dist < best[2]:
                    best = (person["tid"], emb, dist)
            if best is None:
                continue
            tid, emb, dist = best
            strong = dist < self._spatial_threshold * 0.8
            if not self._stay_confirm(role, tid, timestamp, strong):
                continue
            if not self._role_is_best(emb, role):
                continue
            self.locked[role] = int(tid)
            self.pending.pop(role, None)
            logger.info("锁定角色 %s ← tid=%s (距离 %.3f)", role, tid, dist)

    def _stay_confirm(self, role: str, tid: int, timestamp: float, strong: bool) -> bool:
        """驻留确认：在场时长与合格次数双双达标才算通过.

        Args:
            role: 角色名.
            tid: 候选 tid.
            timestamp: 当前时间.
            strong: 是否强匹配（阈值更短）.

        Returns:
            达标返回 True.
        """
        need_time = self._strong_time if strong else self._weak_time
        need_passes = self._strong_passes if strong else self._weak_passes
        pending = self.pending.get(role)
        if pending is not None:
            gap = timestamp - float(pending["last"])
            if gap > self._grace:
                pending = None
            elif pending["tid"] != tid:
                # 同一人换了追踪 ID：继承已累计时长（ID 抖动容忍）
                pending["tid"] = int(tid)
        if pending is None:
            pending = {"tid": int(tid), "acc": 0.0, "passes": 0, "last": timestamp}
            self.pending[role] = pending
        else:
            pending["acc"] = float(pending["acc"]) + max(0.0, timestamp - float(pending["last"]))
        pending["passes"] = int(pending["passes"]) + 1
        pending["last"] = timestamp
        return float(pending["acc"]) >= need_time and int(pending["passes"]) >= need_passes

    def _role_is_best(self, embedding: np.ndarray, role: str) -> bool:
        """竞争唯一性：该脸必须最像目标角色（或差距在 margin 内）."""
        dists: Dict[str, float] = {}
        for other in ROLES:
            tpl = self.template_matrix(other)
            if tpl is None:
                continue
            dists[other] = face_distance(embedding, tpl, self._agg, self._topk)
        if len(dists) < 2 or role not in dists:
            return True
        best_role = min(dists, key=dists.get)
        if best_role == role:
            return True
        return (dists[role] - dists[best_role]) <= self._margin

    def _assign_by_elimination(
        self, timestamp: float, people: List[dict], present_tids: set
    ) -> None:
        """排除法补位：恰好 3 人、恰好 1 个角色空缺、唯一未分配者站在该角色工位内."""
        if len(present_tids) != 3:
            self._elim_streak.clear()
            return
        unlocked = [r for r in ROLES if self.locked.get(r) is None]
        if len(unlocked) != 1:
            self._elim_streak.clear()
            return
        role = unlocked[0]
        assigned = {t for t in self.locked.values() if t is not None}
        free = [p for p in people if p["tid"] not in assigned]
        if len(free) != 1:
            self._elim_streak.clear()
            return
        person = free[0]
        polygon = self._polygons.get(role)
        if not polygon or not person.get("center") or not point_in_polygon(person["center"], polygon):
            self._elim_streak.clear()
            return
        streak = self._elim_streak.get(role) or {"tid": person["tid"], "frames": 0}
        if streak.get("tid") != person["tid"]:
            streak = {"tid": person["tid"], "frames": 0}
        streak["frames"] = int(streak["frames"]) + 1
        self._elim_streak[role] = streak
        if streak["frames"] >= self._elim_frames:
            self.locked[role] = int(person["tid"])
            self.pending.pop(role, None)
            self._elim_streak.pop(role, None)
            logger.info(
                "排除法补位 %s ← tid=%s（连续 %d 帧）",
                role, person["tid"], streak["frames"],
            )

    def _tick_pending(self, timestamp: float, present_tids: set) -> None:
        """每帧巡检驻留确认：已锁定 / 不在画面 / 超时 → 丢弃."""
        for role in list(self.pending.keys()):
            pending = self.pending.get(role)
            if pending is None:
                continue
            if self.locked.get(role) == pending["tid"]:
                self.pending.pop(role, None)
            elif pending["tid"] not in present_tids or timestamp - float(pending["last"]) > self._grace:
                self.pending.pop(role, None)
