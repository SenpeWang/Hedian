"""MOT 跟踪可视化模块：在视频帧上绘制人员边界框、全局 ID 与身份标签."""
from typing import Dict, List, Optional
import cv2
import numpy as np

from modules.tracker.multi_object_tracker import STrack

IDENTITY_COLORS: Dict[str, tuple] = {
    "LEADER": (0, 0, 255),    # 红色 (BGR)
    "ROAD1": (0, 255, 0),     # 绿色
    "ROAD2": (0, 200, 0),     # 深绿色
}
DEFAULT_COLOR: tuple = (0, 255, 0)


def draw_tracks(
    frame: np.ndarray,
    tracks: List[STrack],
    identity_map: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """在视频帧上绘制目标跟踪框与人员身份标识.

    Args:
        frame (np.ndarray): 原始视频帧 (BGR 格式).
        tracks (List[STrack]): 当前帧中的跟踪目标列表.
        identity_map (Optional[Dict[str, int]]): 身份到 track_id 的映射字典，可选.

    Returns:
        np.ndarray: 绘制完成的可视化视频帧.
    """
    annotated_frame = frame.copy()
    for track in tracks:
        identity = track.identity
        if not identity and identity_map:
            for identity_name, mapped_tid in identity_map.items():
                if track.track_id == mapped_tid:
                    identity = identity_name
                    break

        box_color = IDENTITY_COLORS.get(identity, DEFAULT_COLOR)
        x_min, y_min, x_max, y_max = map(int, track.bbox)
        cv2.rectangle(
            annotated_frame,
            (x_min, y_min),
            (x_max, y_max),
            box_color,
            2,
        )

        if identity:
            display_label = f"{identity}(ID:{track.track_id})"
        else:
            display_label = f"ID:{track.track_id}"
        cv2.putText(
            annotated_frame,
            display_label,
            (x_min, max(20, y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2,
        )
    return annotated_frame
