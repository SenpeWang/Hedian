"""
多目标跟踪模块

负责：
- 目标检测（YOLO）
- 多目标跟踪（OC-SORT + ByteTrack）
- 举手检测

使用方式：
    from modules.tracker import TrackerModule
    module = TrackerModule(event_bus, config, paths, display_buffer)
    module.start(video_path, run_id)
"""
from modules.tracker.tracker_module import TrackerModule
from modules.tracker.object_detector import ObjectDetector
from modules.tracker.multi_object_tracker import MultiObjectTracker
from modules.tracker.storage import TrackerStorage

__all__ = [
    "TrackerModule",
    "ObjectDetector",
    "MultiObjectTracker",
    "TrackerStorage",
]
