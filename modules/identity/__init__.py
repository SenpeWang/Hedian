"""人员身份与在岗监控模块（独立进程运行）：检测 + 跟踪 + 人脸身份锁定 + 在岗/离岗判定."""

from modules.identity.face_recognizer import FaceRecognizer, face_distance
from modules.identity.identity_module import IdentityModule
from modules.identity.role_lock import RoleLock
from modules.identity.station_state import StationMonitor, point_in_polygon
from modules.identity.storage_identity import IdentityStorage

__all__ = [
    "IdentityModule",
    "RoleLock",
    "FaceRecognizer",
    "face_distance",
    "StationMonitor",
    "point_in_polygon",
    "IdentityStorage",
]
