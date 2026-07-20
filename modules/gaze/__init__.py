"""
注视检测模块

独立实现，Tracker调用。
负责：头部检测、注视推断、ROI分类、可视化、推送推理结果。

使用方式：
    from modules.gaze.gaze_module import GazeModule
    module = GazeModule(head_model_path, gaze_model_path, roi_json_path, config, display_fn)
    vis_frame = processor.process_frame(frame, ts, frame_count)
"""
from modules.gaze.gaze_module import GazeModule
from modules.gaze.head_detector import HeadDetector
from modules.gaze.gaze_estimator import GazeEstimator
from modules.gaze.roi_classifier import ROIClassifier
from modules.gaze.storage_gaze import GazeStorage

__all__ = [
    "GazeModule",
    "HeadDetector",
    "GazeEstimator",
    "ROIClassifier",
    "GazeStorage",
]
