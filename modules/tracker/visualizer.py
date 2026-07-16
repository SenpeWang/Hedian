"""
MOT 可视化模块
"""
import cv2
import numpy as np

ROLE_COLORS = {
    "LEADER": (0, 0, 255),
    "ROAD1":  (0, 255, 0),
    "ROAD2":  (0, 200, 0),
}
DEFAULT_COLOR = (0, 255, 0)


def draw_tracks(frame, tracks, roles=None):
    vis = frame.copy()
    for track in tracks:
        role = None
        if roles:
            for role_name, tid in roles.items():
                if track.track_id == tid:
                    role = role_name
                    break
        color = ROLE_COLORS.get(role, DEFAULT_COLOR)
        x1, y1, x2, y2 = map(int, track.bbox)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = role if role else f"ID:{track.track_id}"
        cv2.putText(vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def draw_supervision(frame, leader_pos, target_pos, distance, state):
    vis = frame.copy()
    if leader_pos is None or target_pos is None:
        return vis
    if distance <= 280:
        color = (0, 255, 0)
    elif distance <= 400:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)
    lc = tuple(map(int, leader_pos))
    tc = tuple(map(int, target_pos))
    cv2.line(vis, lc, tc, color, 3)
    mid = ((lc[0] + tc[0]) // 2, (lc[1] + tc[1]) // 2)
    cv2.putText(vis, f"{int(distance)}px", mid, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    label = f"{state} {int(distance)}px"
    overlay = vis.copy()
    cv2.rectangle(overlay, (5, 5), (350, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
    cv2.putText(vis, label, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return vis
