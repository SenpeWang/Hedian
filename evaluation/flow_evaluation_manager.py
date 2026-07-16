"""
流程评估编排模块

负责编排流程评估的完整流程：
1. 接收流程事件
2. 调用数据提取器获取事件
3. 调用评估模块进行评估
4. 保存评估结果
5. 推送评估报告到前端
"""
import os
import json
import logging
import threading
import concurrent.futures
import redis
from typing import Dict, List, Optional, Callable

from core.event_bus import EventBus, EventTopic
from evaluation.flow_data_extractor import FlowDataExtractor
from evaluation.qwen_evaluator import QwenEvaluator

logger = logging.getLogger("evaluation.manager")


class FlowEvaluationManager:
    """
    流程评估编排器

    负责编排流程评估的完整流程。
    """

    def __init__(
        self,
        event_bus: EventBus,
        result_dir: str,
        fps: float = 30.0,
        model_path: str = None,
        display_fn: Callable = None,
    ):
        """
        初始化评估编排器

        Args:
            event_bus: 消息总线
            result_dir: 结果目录
            fps: 帧率
            model_path: Qwen 模型路径
            display_fn: 推送事件函数
        """
        self._event_bus = event_bus
        self._result_dir = result_dir
        self._fps = fps
        self._display_fn = display_fn

        # 默认模型路径
        if model_path is None:
            from pathlib import Path
            model_path = str(Path(__file__).parent.parent / "models" / "evaluation")

        # 数据提取器（传入 Redis 客户端用于读取模块进度）
        redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        self._data_extractor = FlowDataExtractor(result_dir, redis_client=redis_client)

        # 评估器
        self._qwen_evaluator = QwenEvaluator(model_path=model_path)

        # 流程管理
        self._active_flows = {}
        self._completed_flows = []
        self._segment_reports = []

        self._lock = threading.Lock()

        # 异步评估
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._eval_futures = {}
        self._eval_lock = threading.Lock()

        # 结果目录
        self._qwen_dir = os.path.join(result_dir, "qwen")
        os.makedirs(self._qwen_dir, exist_ok=True)

        # 订阅 event_bus 事件
        event_bus.subscribe(EventTopic.FLOW_STARTED, self._on_flow_started)
        event_bus.subscribe(EventTopic.FLOW_ENDED, self._on_flow_ended)

        logger.info("FlowEvaluationManager 初始化完成")

    def _on_flow_started(self, msg: dict) -> None:
        """处理流程开始事件"""
        data = msg.get("data", {})
        flow_id = data.get("flow_id", 0)
        flow_type = data.get("flow_type", "unknown")
        ts = data.get("flow_start_sec", msg.get("ts", 0))

        flow = {
            "flow_id": flow_id,
            "flow_type": flow_type,
            "flow_start_sec": ts,
            "start_source": data.get("start_source", "unknown"),
            "flow_end_sec": None,
            "end_source": None,
            "device_code": data.get("device_code", ""),
            "content_checklist": data.get("content_checklist", {}),
            "sub_flows": [],
        }

        with self._lock:
            self._active_flows[flow_id] = flow

        if self._display_fn:
            self._display_fn("flow_start", {
                "localSec": ts,
                "flowId": flow_id,
                "flow_type": flow_type,
                "flow_start_sec": ts,
                "start_source": data.get("start_source", "unknown"),
            })

        logger.info(f"流程开始 flow_id={flow_id} type={flow_type} @{ts:.1f}s")

    def _on_flow_ended(self, msg: dict) -> None:
        """处理流程结束事件"""
        data = msg.get("data", {})
        flow_id = data.get("flow_id", 0)
        ts = data.get("flow_end_sec", msg.get("ts", 0))

        with self._lock:
            flow = self._active_flows.pop(flow_id, None)
            if flow is None:
                flow = {
                    "flow_id": flow_id,
                    "flow_type": data.get("flow_type", "unknown"),
                    "flow_start_sec": data.get("flow_start_sec", 0),
                    "start_source": data.get("start_source", "unknown"),
                }

            flow["flow_end_sec"] = ts
            flow["end_source"] = data.get("end_source", "unknown")
            flow["flow_continue_sec"] = data.get(
                "flow_continue_sec",
                round(ts - flow.get("flow_start_sec", 0), 2),
            )
            flow["content_checklist"] = data.get("content_checklist", {})
            flow["device_code"] = data.get("device_code", "")
            self._completed_flows.append(flow)

        # 持久化流程事件
        self._save_flow_events()

        if self._display_fn:
            self._display_fn("flow_end", {
                "localSec": ts,
                "flowId": flow_id,
                "flow_type": flow.get("flow_type"),
                "flow_start_sec": flow.get("flow_start_sec"),
                "flow_end_sec": ts,
                "flow_continue_sec": flow.get("flow_continue_sec"),
                "end_source": data.get("end_source"),
                "content_checklist": flow.get("content_checklist"),
            })

        logger.info(f"流程结束 flow_id={flow_id} @{ts:.1f}s")

        # 触发评估
        self._evaluate_and_save(flow)

    def _evaluate_and_save(self, flow: Dict) -> None:
        """
        触发评估并保存结果

        Args:
            flow: 流程数据
        """
        start_sec = flow.get("flow_start_sec", 0)
        end_sec = flow.get("flow_end_sec") or start_sec

        # 使用数据提取器加载事件
        voice_events, tracker_events, gaze_events = self._data_extractor.extract(
            start_sec, end_sec, wait=True, timeout=300
        )

        flow_data = {
            "flow_id": flow.get("flow_id"),
            "flow_type": flow.get("flow_type"),
            "flow_start_sec": start_sec,
            "start_source": flow.get("start_source", "unknown"),
            "flow_end_sec": end_sec,
            "end_source": flow.get("end_source", "unknown"),
            "flow_continue_sec": flow.get(
                "flow_continue_sec",
                round(end_sec - start_sec, 2),
            ),
            "voice_events": voice_events,
            "tracker_events": tracker_events,
            "gaze_events": gaze_events,
        }

        # 保存提取并拼接的 JSON 文件到 evaluation 文件夹内
        self._data_extractor.save_extracted_data(flow_data)

        if flow.get("flow_type") in ("supervision", "info_notice"):
            flow_data["content_checklist"] = flow.get("content_checklist", {})
        if flow.get("flow_type") == "self_ticket":
            flow_data["device_code"] = flow.get("device_code", "")

        # 异步评估
        future = self._executor.submit(self._do_evaluate, flow_data)
        with self._eval_lock:
            self._eval_futures[flow["flow_id"]] = future
        future.add_done_callback(
            lambda f, fid=flow["flow_id"]: self._on_eval_done(f, fid)
        )

    def _do_evaluate(self, flow_data: Dict) -> tuple:
        """
        执行评估

        Args:
            flow_data: 流程数据

        Returns:
            (评估结果, 流程数据)
        """
        flow_id = flow_data["flow_id"]
        eval_local_sec = flow_data.get("flow_end_sec", flow_data.get("flow_start_sec", 0))

        def stream_cb(text_chunk):
            if self._display_fn:
                self._display_fn("segment_report_stream", {
                    "localSec": eval_local_sec,
                    "flowId": flow_id,
                    "chunk": text_chunk,
                })

        try:
            # Qwen 大模型评估
            qwen_report = self._qwen_evaluator.evaluate(
                flow_data,
                stream_callback=stream_cb,
            )

            # 结果
            report = {
                "flow_type": qwen_report.get("flow_type", "未知"),
                "score": qwen_report.get("score", 0),
                "report_text": qwen_report.get("report_text", ""),
                "prompt": qwen_report.get("prompt", ""),
            }

        except Exception as e:
            flow_type_cn = "监护制" if flow_data.get("flow_type") == "supervision" else (
                "信息通报" if flow_data.get("flow_type") == "info_notice" else "自唱票"
            )
            report = {
                "flow_type": flow_type_cn,
                "score": 0,
                "report_text": f"评估失败: {e}",
                "prompt": "",
            }

        return report, flow_data

    def is_evaluating(self) -> bool:
        """是否有正在评估的流程"""
        with self._eval_lock:
            return len(self._eval_futures) > 0

    def _on_eval_done(self, future: concurrent.futures.Future, flow_id: int) -> None:
        """评估完成回调"""
        with self._eval_lock:
            self._eval_futures.pop(flow_id, None)

        try:
            report, flow_data = future.result()
        except Exception as e:
            report = {
                "flow_type": "未知",
                "score": 0,
                "report_text": f"评估失败: {e}",
                "rule_score": 0,
                "rule_report": "",
                "prompt": "",
            }
            flow_data = {"flow_continue_sec": 0}

        with self._lock:
            self._segment_reports.append(report)

        self._save_segment_reports()

        if self._display_fn:
            self._display_fn("segment_report", {
                "localSec": flow_data.get(
                    "flow_end_sec",
                    flow_data.get("flow_start_sec", 0),
                ),
                "flowId": flow_data.get("flow_id"),
                "flow_type": report.get("flow_type"),
                "score": report.get("score", 0),
                "report_text": report.get("report_text", ""),
                "flow_continue_sec": flow_data.get("flow_continue_sec"),
            })

    def _save_flow_events(self) -> None:
        """保存流程事件（开始/结束时间）"""
        path = os.path.join(self._result_dir, "flow_events.json")

        with self._lock:
            flows = list(self._completed_flows)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(flows, f, ensure_ascii=False, indent=2)
            logger.info(f"保存 {len(flows)} 个流程事件")

        except Exception as e:
            logger.error(f"保存流程事件失败: {e}", exc_info=True)

    def _save_segment_reports(self) -> None:
        """保存分段评估报告"""
        path = os.path.join(self._qwen_dir, "Qwen_segment_reports.json")

        with self._lock:
            reports = list(self._segment_reports)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(reports, f, ensure_ascii=False, indent=2)
            logger.info(f"保存 {len(reports)} 个分段评估报告")

        except Exception as e:
            logger.error(f"保存分段评估报告失败: {e}", exc_info=True)

    def finalize(self) -> None:
        """视频结束时的清理"""
        with self._lock:
            for flow_id, flow in list(self._active_flows.items()):
                flow["flow_end_sec"] = flow.get("flow_start_sec", 0) + 60.0
                flow["end_source"] = "finalize"
                self._completed_flows.append(flow)
            self._active_flows.clear()

        # 持久化流程事件
        self._save_flow_events()

        for flow in self._completed_flows:
            already = any(
                r.get("flow_id") == flow["flow_id"]
                for r in self._segment_reports
            )
            if not already:
                self._evaluate_and_save(flow)

        with self._eval_lock:
            pending = list(self._eval_futures.values())
        if pending:
            concurrent.futures.wait(pending, timeout=300)
