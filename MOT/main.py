#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标跟踪处理器 - 模块化版本
"""
import os
import sys
import cv2
import json
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

from config.settings import Settings
from detector.detector import ObjectDetector
from tracker.tracker import DynamicTracker
from utils.visualizer import Visualizer


class MultiObjectTracker:
    """多目标跟踪处理器"""
    
    def __init__(self):
        self.settings = Settings()
        self._check_environment()
        
        self.detector = ObjectDetector(
            model_path=self.settings.detection.model_path,
            conf_threshold=self.settings.detection.conf_threshold,
            nms_threshold=self.settings.detection.nms_threshold,
            img_size=self.settings.detection.img_size
        )
        
        self.tracker = DynamicTracker()
        self.visualizer = Visualizer()
        
        self.output_dir = self.settings.path.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _check_environment(self):
        if not os.path.exists(self.settings.path.input_video):
            raise FileNotFoundError(f"视频不存在: {self.settings.path.input_video}")
        if not os.path.exists(self.settings.detection.model_path):
            raise FileNotFoundError(f"YOLO模型不存在: {self.settings.detection.model_path}")
    
    def process_video(self, max_frames: int = None) -> Dict:
        """处理视频"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_video = os.path.join(self.output_dir, f"result_{timestamp}.mp4")
        output_json = os.path.join(self.output_dir, f"result_{timestamp}.json")
        
        cap = cv2.VideoCapture(self.settings.path.input_video)
        if not cap.isOpened():
            return {'error': '无法打开视频'}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        actual_frames = min(total_frames, max_frames) if max_frames else total_frames
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        frame_count = 0
        tracking_results = []

        pbar = tqdm(total=actual_frames, desc="处理视频", unit="帧")

        try:
            while frame_count < actual_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detector.detect_with_pose(frame)
                tracks = self.tracker.track(frame, detections)

                vis_frame = self.visualizer.draw_tracks(frame, tracks)
                out.write(vis_frame)

                tracking_results.append({
                    'frame_id': frame_count,
                    'tracks': [{'id': t.track_id, 'bbox': t.bbox.tolist(), 'score': float(t.score)} for t in tracks]
                })

                frame_count += 1
                track_ids = ','.join([str(t.track_id) for t in tracks]) if tracks else '无'
                pbar.set_postfix_str(f"ID:{track_ids}")
                pbar.update(1)

        finally:
            pbar.close()
            cap.release()
            out.release()
        
        stats = self.tracker.get_statistics()
        
        result = {
            'status': 'completed',
            'tracker_type': 'dynamic',
            'frame_count': frame_count,
            'max_id_reached': stats.get('max_id_generated', stats.get('next_id', 0) - 1),
            'id_switches': stats.get('id_switch_count', 0),
            'output_video': output_video,
            'statistics': stats
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n处理完成:")
        print(f"  帧数: {frame_count}")
        print(f"  最大ID: {result['max_id_reached']}")
        print(f"  ID切换: {result['id_switches']}")
        print(f"  输出: {output_video}")
        
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='多目标跟踪处理器')
    parser.add_argument('--max', type=int, default=None, help='最大处理帧数')
    args = parser.parse_args()
    
    processor = MultiObjectTracker()
    processor.process_video(max_frames=args.max)


if __name__ == "__main__":
    main()
