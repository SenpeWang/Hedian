"""
可视化模块 — 角色标签颜色（不绘制骨架）
"""
import cv2
import numpy as np
from typing import List, Dict

# 角色颜色 BGR
ROLE_COLORS = {
    "LEADER": (0, 0, 255),    # 红
    "ROAD1":  (0, 255, 0),    # 绿
    "ROAD2":  (0, 255, 0),    # 绿
}
DEFAULT_COLOR = (0, 255, 0)


class Visualizer:
    def __init__(self):
        pass

    def draw_tracks(self, frame: np.ndarray, tracks: List,
                    show_id: bool = True, show_conf: bool = False) -> np.ndarray:
        for track in tracks:
            role = getattr(track, "role", None)
            color = ROLE_COLORS.get(role, DEFAULT_COLOR)

            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if show_id:
                label = role if role else f"ID:{track.track_id}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if show_conf:
                cv2.putText(frame, f"{track.score:.2f}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def draw_info(self, frame: np.ndarray, info: Dict) -> np.ndarray:
        y = 30
        for k, v in info.items():
            cv2.putText(frame, f"{k}: {v}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 30
        return frame
