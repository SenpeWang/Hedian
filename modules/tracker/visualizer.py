"""MOT 跟踪可视化模块：在视频帧上绘制人员边界框、全局 ID 与身份标签."""

from typing import Dict, List, Optional

import cv2
import numpy as np

from modules.tracker.multi_object_tracker import STrack

IDENTITY_COLORS: Dict[str, tuple] = {
    "LEADER": (0, 0, 255),  # 红色 (BGR)
    "ROAD1": (0, 255, 0),  # 绿色
    "ROAD2": (0, 200, 0),  # 深绿色
}
DEFAULT_COLOR: tuple = (0, 255, 0)
# 离岗叠加层配色：已确认离岗=红色告警，防抖期内离开=橙色预警
AWAY_COLOR: tuple = (0, 0, 255)
LEAVING_COLOR: tuple = (0, 165, 255)


def _status_overlay(station: Optional[dict]) -> Optional[str]:
    """由在岗快照生成头顶状态文案.

    Args:
        station: 单个身份的状态快照，含 away / away_sec 字段，可为 None.

    Returns:
        需要叠加的文案；在岗或数据缺失时返回 None.
    """
    if not station:
        return None
    away_sec = float(station.get("away_sec") or 0.0)
    # cv2.putText 不支持中文（渲染成 ?），此处统一用 ASCII
    if station.get("away"):
        return f"AWAY {away_sec:.1f}s"
    if away_sec > 0.0:
        return f"LEAVE {away_sec:.1f}s"
    return None


def draw_tracks(
    frame: np.ndarray,
    tracks: List[STrack],
    identity_map: Optional[Dict[str, int]] = None,
    station_map: Optional[Dict[str, dict]] = None,
) -> np.ndarray:
    """在视频帧上绘制目标跟踪框与人员身份标识.

    Args:
        frame: 原始视频帧 (BGR 格式).
        tracks: 当前帧中的跟踪目标列表.
        identity_map: 身份到 track_id 的映射字典，可选.
        station_map: 身份到在岗快照的映射（含 away / away_sec），可选；
            提供时离岗目标框变红并在头顶叠加离岗倒计时.

    Returns:
        绘制完成的可视化视频帧.
    """
    annotated_frame = frame.copy()
    for track in tracks:
        identity = track.identity
        if not identity and identity_map:
            for identity_name, mapped_track_id in identity_map.items():
                if track.track_id == mapped_track_id:
                    identity = identity_name
                    break

        station = (station_map or {}).get(identity) if identity else None
        is_away = bool(station and station.get("away"))
        is_leaving = bool(
            station
            and not is_away
            and float(station.get("away_sec") or 0.0) > 0.0
        )

        if is_away:
            box_color = AWAY_COLOR
        elif is_leaving:
            box_color = LEAVING_COLOR
        else:
            box_color = IDENTITY_COLORS.get(identity, DEFAULT_COLOR)

        x_min, y_min, x_max, y_max = map(int, track.bbox)
        cv2.rectangle(
            annotated_frame,
            (x_min, y_min),
            (x_max, y_max),
            box_color,
            3 if is_away else 2,
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

        overlay = _status_overlay(station)
        if overlay:
            # 头顶第二行显示离岗/离开时长；贴到画面顶端时改画进框内顶部
            overlay_y = y_min - 34
            if overlay_y < 20:
                overlay_y = min(y_max - 10, y_min + 26)
            cv2.putText(
                annotated_frame,
                overlay,
                (x_min, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2,
            )
    return annotated_frame
