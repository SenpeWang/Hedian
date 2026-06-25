"""
多目标跟踪模块

负责：
- 目标检测（YOLO）
- 多目标跟踪（OC-SORT + ByteTrack）
- 举手检测

使用方式：
    from modules.mot import MOTModule
    module = MOTModule(bus, config, paths, aggregator)
    module.start(video_path, run_id)
"""
from modules.mot.mot_module import MOTModule
from modules.mot.object_detector import ObjectDetector
from modules.mot.multi_object_tracker import MultiObjectTracker
from modules.mot.hand_raiser import HandRaiser
from modules.mot.result_storage import MOTResultStorage

__all__ = [
    "MOTModule",
    "ObjectDetector",
    "MultiObjectTracker",
    "HandRaiser",
    "MOTResultStorage",
]
