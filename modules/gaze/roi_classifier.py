"""
ROI 分类模块

负责加载 ROI 配置，分类注视方向。
"""
import os
import json
import logging
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2

from modules.gaze.head_detector import HeadBox

logger = logging.getLogger("module.gaze.roi")


class ROIClassifier:
    """
    ROI 分类器

    负责加载 ROI 配置，分类注视方向。
    """

    def __init__(
        self,
        roi_json_path: str,
        inout_threshold: float = 0.5,
        heatmap_threshold: float = 0.3,
    ):
        """
        初始化 ROI 分类器

        Args:
            roi_json_path: ROI 配置文件路径
            inout_threshold: 进出阈值
            heatmap_threshold: 热力图阈值
        """
        self._inout_threshold = inout_threshold
        self._heatmap_threshold = heatmap_threshold

        # 加载 ROI 配置
        self._gaze_rois, self._head_zones = self._load_roi_polygons(roi_json_path)

        logger.info(f"加载 ROI 配置: {len(self._gaze_rois)} 个 ROI 区域")

    def _load_roi_polygons(
        self, roi_json_path: str
    ) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]:
        """
        加载 ROI 多边形（支持 LabelMe 格式）

        Args:
            roi_json_path: ROI 配置文件路径

        Returns:
            (gaze_rois, head_zones)
        """
        try:
            with open(roi_json_path, encoding="utf-8") as f:
                data = json.load(f)

            gaze_rois = []
            head_zones = []

            # 支持 LabelMe 格式 (shapes 数组)
            shapes = data.get("shapes", data) if isinstance(data, dict) else data

            for item in shapes:
                if not isinstance(item, dict):
                    continue

                group_id = item.get("group_id", 1)
                label = item.get("label", "")
                points = item.get("points", [])
                shape_type = item.get("shape_type", "polygon")

                if not points:
                    continue

                # 处理圆形
                if shape_type == "circle":
                    cx, cy = points[0]
                    ex, ey = points[1]
                    radius = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
                    angles = [i * 2 * math.pi / 64 for i in range(64)]
                    points = [[cx + radius * math.cos(a), cy + radius * math.sin(a)] for a in angles]

                contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

                if group_id == 1:
                    gaze_rois.append((label, contour))
                elif group_id in (2, 3):
                    head_zones.append((label, contour))

            return gaze_rois, head_zones

        except Exception as e:
            logger.error(f"加载 ROI 配置失败: {e}", exc_info=True)
            return [], []

    def filter_heads_by_zone(self, heads: List[HeadBox]) -> List[HeadBox]:
        """
        只保留在 head_zone 内的头部

        Args:
            heads: 头部边界框列表

        Returns:
            过滤后的头部边界框列表
        """
        if not self._head_zones:
            return heads

        filtered = []
        for head in heads:
            for _, contour in self._head_zones:
                if cv2.pointPolygonTest(contour, (float(head.cx), float(head.cy)), False) >= 0:
                    filtered.append(head)
                    break

        return filtered

    def extract_gaze_point(
        self,
        heatmap_2d: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> Optional[Tuple[int, int]]:
        """
        从热力图提取注视点（加权质心）

        Args:
            heatmap_2d: 热力图
            img_w: 图像宽度
            img_h: 图像高度

        Returns:
            注视点坐标 (x, y)，或 None
        """
        mask = heatmap_2d > self._heatmap_threshold

        if not np.any(mask):
            peak_idx = np.argmax(heatmap_2d)
            py, px = np.unravel_index(peak_idx, heatmap_2d.shape)
            gx = int(px / heatmap_2d.shape[1] * img_w)
            gy = int(py / heatmap_2d.shape[0] * img_h)
            return gx, gy

        weights = heatmap_2d[mask]
        ys, xs = np.where(mask)
        gx = int(np.average(xs, weights=weights) / heatmap_2d.shape[1] * img_w)
        gy = int(np.average(ys, weights=weights) / heatmap_2d.shape[0] * img_h)

        return gx, gy

    def classify_gaze(
        self,
        inout_score: float,
        gaze_point: Tuple[int, int],
    ) -> Tuple[str, str]:
        """
        分类注视状态

        Args:
            inout_score: 进出分数
            gaze_point: 注视点坐标

        Returns:
            (status, roi_label)
        """
        if inout_score < self._inout_threshold:
            return "OUTSIDE_FRAME", ""

        gx, gy = gaze_point

        for label, contour in self._gaze_rois:
            dist = cv2.pointPolygonTest(contour, (float(gx), float(gy)), False)
            if dist >= 0:
                return "IN_ROI", label

        return "OUTSIDE_ROI", ""
