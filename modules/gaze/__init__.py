"""
注视检测模块

负责：
- 头部检测（YOLOv8）
- 注视推断（Gazelle）
- ROI 分类

使用方式：
    from modules.gaze import GazeModule
    module = GazeModule(bus, config, paths, aggregator)
    module.start(video_path, run_id)
"""
from modules.gaze.gaze_module import GazeModule
from modules.gaze.head_detector import HeadDetector
from modules.gaze.gaze_estimator import GazeEstimator
from modules.gaze.roi_classifier import ROIClassifier

__all__ = [
    "GazeModule",
    "HeadDetector",
    "GazeEstimator",
    "ROIClassifier",
]
