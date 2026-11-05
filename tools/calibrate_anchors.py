# -*- coding: utf-8 -*-
"""工位 anchor 自动标定.

用身份锁定期间的人体中心中位数作为工位基准点(anchor)，替代手点或几何质心。
anchor 只影响 MultiObjectTracker.assign_identities() 的最近距离分配
（人脸锁定启用时它只是兜底路径），但仍然要准，否则回退时会分错人。

用法::

    python tools/calibrate_anchors.py --video ../data/raw/Nuclear/camFRONT2.mpg --room 2 --frames 900
    python tools/calibrate_anchors.py --video data/videos/camFRONT.mpg --room 1

输出可直接粘贴进 data/workstations.json 的 `rooms.<room>.<身份>.anchor`。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from core.config_manager import ConfigManager  # noqa: E402
from core.path_manager import PathConfig  # noqa: E402
from modules.identity.identity_monitor import IdentityMonitor  # noqa: E402
from modules.identity.role_lock import ROLES  # noqa: E402
from modules.tracker.multi_object_tracker import MultiObjectTracker  # noqa: E402
from modules.tracker.object_detector import ObjectDetector  # noqa: E402
from modules.tracker.scene_config import load_scene  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="工位 anchor 自动标定")
    parser.add_argument("--video", required=True, help="标定用视频路径")
    parser.add_argument("--room", type=str, default="1", help="机室编号")
    parser.add_argument("--frames", type=int, default=900, help="采样帧数上限")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件")
    return parser.parse_args()


def main() -> None:
    """跑视频，按身份锁定结果统计各角色中心点中位数."""
    args = parse_args()
    config = ConfigManager(os.path.join(BASE_DIR, args.config)).to_dict()
    config.setdefault("scene", {})["room_id"] = args.room

    paths = PathConfig.from_config(config)
    scene = load_scene(config, paths.base_dir)

    detection_config = dict(config.get("tracker", {}).get("detection", {}))
    detection_config.update(config.get("identity", {}).get("detection", {}) or {})
    detector = ObjectDetector(
        model_path=paths.get_model_path(
            "detection", detection_config.get("weights", "yolo11_MOT.pt")
        ),
        pose_model_path=None,
        conf_threshold=float(detection_config.get("conf_threshold", 0.65)),
        nms_threshold=float(detection_config.get("nms_threshold", 0.35)),
        img_size=int(detection_config.get("img_size", 640)),
    )
    tracker = MultiObjectTracker(workstations=scene.anchors or None)

    identity_config = config.get("identity", {}) or {}
    monitor = IdentityMonitor(
        scene,
        face_config=identity_config.get("face", {}) or {},
        lock_config=identity_config.get("lock", {}) or {},
        away_hold_sec=float(identity_config.get("away_hold_sec", 2.0)),
    )
    if not monitor.initialize():
        print("[calib] 身份监控初始化失败")
        return

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print(f"[calib] 无法打开视频: {args.video}")
        return
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)

    samples = {role: [] for role in ROLES}
    frame_count = 0
    while frame_count < args.frames:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = frame_count / fps
        frame_count += 1

        high_det, low_det = detector.detect_two_thresholds(frame)
        tracks = tracker.track(frame, high_det + low_det)
        locked = monitor.process_frame(frame, tracks, timestamp)

        tid_to_center = {int(t.track_id): t.get_center() for t in tracks}
        for role, tid in locked.items():
            if tid is None:
                continue
            center = tid_to_center.get(int(tid))
            if center is not None:
                samples[role].append((float(center[0]), float(center[1])))

    capture.release()

    print(f"[calib] 机室={args.room} 视频={args.video} 处理帧数={frame_count}")
    anchors = {}
    for role in ROLES:
        pts = samples[role]
        if not pts:
            print(f"[calib] {role}: 无样本（人脸未锁定？）")
            continue
        xs = statistics.median(p[0] for p in pts)
        ys = statistics.median(p[1] for p in pts)
        anchors[role] = [int(round(xs)), int(round(ys))]
        print(f"[calib] {role}: {len(pts)} 样本 -> anchor {anchors[role]}")
    print("[calib] 可粘贴进 data/workstations.json:")
    print(json.dumps({"rooms": {args.room: {k: {"anchor": v} for k, v in anchors.items()}}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
