"""
MOT 可视化模块
"""
import cv2

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


