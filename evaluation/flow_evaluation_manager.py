"""流程评估编排与生命周期管理模块.

负责流程事件监听、多模态事实数据异步提取、调用大模型进行异步推理，
并在前端画面播放到达流程结束时刻时精准通过 WebSocket 文本通道进行流式直推。
"""
import concurrent.futures
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import redis

from core.event_bus import EventStream, EventTopic
from evaluation.flow_data_extractor import FlowDataExtractor
from evaluation.qwen_evaluator import QwenEvaluator

logger = logging.getLogger("evaluation.manager")


class FlowEvaluationManager:
    """流程评估编排与生命周期管理器.

    统管多模态数据提取、Qwen 大模型异步推理与流式推流。
    完全与视频帧打包中间件物理解耦，仅依据前端播放时钟精准触发流式涌现。
    """

    def __init__(
        self,
        event_bus: EventStream,
        result_dir: str,
        fps: float = 30.0,
        model_path: Optional[str] = None,
        sync_fn: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        direct_fn: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        inference_fn: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        get_playback_sec_fn: Optional[Callable[[], float]] = None,
    ):
        """初始化流程评估编排器.

        Args:
            event_bus (EventStream): 全局事件发布订阅总线.
            result_dir (str): 本次运行结果保存目录路径.
            fps (float): 视频基准帧率，默认 30.0.
            model_path (Optional[str]): Qwen 大模型本地权重路径.
            sync_fn (Optional[Callable]): 经过中间件对齐的系统通知推送函数（flow_start/flow_end）.
            direct_fn (Optional[Callable]): 完全绕过中间件的评估流式直推函数（WebSocket 文本通道）.
            inference_fn (Optional[Callable]): 兼容回退推送函数.
            get_playback_sec_fn (Optional[Callable[[], float]]): 获取前端实际画面播放秒数的函数.
        """
        self._event_bus: EventStream = event_bus
        self._result_dir: str = result_dir
        self._fps: float = fps
        self._sync_fn = sync_fn or inference_fn
        self._direct_fn = direct_fn or inference_fn
        self._get_playback_sec_fn = get_playback_sec_fn

        # 默认模型路径
        if model_path is None:
            model_path = "models/evaluation/Qwen3-8B"

        # 多模态事实数据提取器
        redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        self._data_extractor = FlowDataExtractor(result_dir, redis_client=redis_client)

        # Qwen 大模型评估器
        self._qwen_evaluator = QwenEvaluator(model_path=model_path)

        # 流程与评估状态
        self._completed_flows: List[Dict[str, Any]] = []
        self._segment_reports: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # 异步评估线程池（单流程单线程线性流水线执行：等待模块 ➔ 提取 ➔ 评估 ➔ 存盘）
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._eval_futures: Dict[Any, concurrent.futures.Future] = {}
        self._eval_lock = threading.Lock()

        # 评估结果保存子目录
        self._qwen_dir = os.path.join(result_dir, "qwen")
        os.makedirs(self._qwen_dir, exist_ok=True)

        # 订阅流程生命周期事件
        self._event_bus.subscribe(EventTopic.FLOW_STARTED, self._on_flow_started)
        self._event_bus.subscribe(EventTopic.FLOW_ENDED, self._on_flow_ended)

        logger.info("FlowEvaluationManager 初始化完成")

    def set_result_dir(self, result_dir: str) -> None:
        """动态更新评估结果输出目录.

        Args:
            result_dir (str): 新的结果目录路径.
        """
        self._result_dir = result_dir
        self._qwen_dir = os.path.join(result_dir, "qwen")
        os.makedirs(self._qwen_dir, exist_ok=True)
        if hasattr(self, "_data_extractor") and self._data_extractor:
            self._data_extractor._result_dir = result_dir
        logger.info(f"FlowEvaluationManager 结果目录更新为: {result_dir}")

    def reset(self) -> None:
        """重置评估编排器状态，清理历史任务与缓存."""
        with self._lock:
            self._completed_flows.clear()
            self._segment_reports.clear()
        with self._eval_lock:
            for future_item in self._eval_futures.values():
                if not future_item.done():
                    future_item.cancel()
            self._eval_futures.clear()
        logger.info("FlowEvaluationManager 已重置")

    def _on_flow_started(self, msg: Dict[str, Any]) -> None:
        """处理流程开始事件.

        Args:
            msg (Dict[str, Any]): 事件总线通知消息.
        """
        data = msg.get("data", {})
        flow_id = data.get("flow_id", 0)
        flow_type = data.get("flow_type", "unknown")
        timestamp = data.get("flow_start_sec", msg.get("ts", 0.0))

        if self._sync_fn:
            self._sync_fn("flow_start", {
                "localSec": timestamp,
                "tag": "flow_start",
                "data": {
                    "flow_id": flow_id,
                    "flow_type": flow_type,
                    "flow_start_sec": timestamp,
                    "start_source": data.get("start_source", "unknown"),
                },
            })

        logger.info(f"收到流程开始事件 flow_id={flow_id} type={flow_type} @{timestamp:.1f}s")

    def _on_flow_ended(self, msg: Dict[str, Any]) -> None:
        """处理流程结束事件，启动后台异步评估流水线.

        Args:
            msg (Dict[str, Any]): 事件总线通知消息.
        """
        flow = msg.get("data", {})
        flow_id = flow.get("flow_id", 0)
        if not flow_id:
            logger.warning("FLOW_ENDED 事件缺少 flow_id")
            return

        with self._lock:
            self._completed_flows.append(flow)

        if self._sync_fn:
            self._sync_fn("flow_end", {
                "localSec": flow.get("flow_end_sec", 0.0),
                "tag": "flow_end",
                "data": flow,
            })

        logger.info(f"收到流程结束事件 flow_id={flow_id}，提交后台异步评估")

        future = self._executor.submit(self._process_flow_pipeline, flow)
        with self._eval_lock:
            self._eval_futures[flow_id] = future
        future.add_done_callback(
            lambda completed_future, flow_identifier=flow_id: self._on_pipeline_done(completed_future, flow_identifier)
        )

    def _on_pipeline_done(self, future: concurrent.futures.Future, flow_id: int) -> None:
        """后台单流程异步评估完成回调.

        Args:
            future (concurrent.futures.Future): 执行完成的 Future 对象.
            flow_id (int): 流程编号.
        """
        with self._eval_lock:
            self._eval_futures.pop(flow_id, None)

        try:
            future.result()
            logger.info(f"流程 flow_id={flow_id} 异步评估全链路执行完毕")
        except Exception as pipeline_error:
            logger.error(f"流程 flow_id={flow_id} 异步评估出现异常: {pipeline_error}", exc_info=True)

    def _process_flow_pipeline(self, flow: Dict[str, Any]) -> None:
        """后台独立 Worker：等待模块 ➔ 提取事实 ➔ 大模型推理 ➔ 存盘推流.

        Args:
            flow (Dict[str, Any]): 流程基础元数据字典.
        """
        flow_id = int(flow.get("flow_id", 0))
        flow_type = str(flow.get("flow_type", "unknown"))
        start_sec = float(flow.get("flow_start_sec", 0.0))
        end_sec = float(flow.get("flow_end_sec", start_sec))

        # 步骤 1：非阻塞等待各算法模块（语音/跟踪/行为/注视）推理进度均到达 end_sec
        logger.info(f"[flow_id={flow_id}] 检查并等待各算法模块推理进度到达 {end_sec:.2f}s...")
        self._data_extractor._wait_all_modules(end_sec, timeout=90)

        # 步骤 2：提取多模态事实数据
        voice_events, tracker_events, gaze_events, behavior_events = self._data_extractor.extract(
            start_sec, end_sec, wait=False, timeout=90
        )

        flow_data: Dict[str, Any] = {
            "flow_id": flow_id,
            "flow_type": flow_type,
            "flow_start_sec": start_sec,
            "start_source": flow.get("start_source", "unknown"),
            "flow_end_sec": end_sec,
            "end_source": flow.get("end_source", "unknown"),
            "flow_continue_sec": flow.get("flow_continue_sec", round(end_sec - start_sec, 2)),
            "voice_events": voice_events,
            "tracker_events": tracker_events,
            "gaze_events": gaze_events,
            "behavior_events": behavior_events,
        }

        if flow_type in ("supervision", "info_notice"):
            flow_data["content_checklist"] = flow.get("content_checklist", {})
        if flow_type == "self_ticket":
            flow_data["device_code"] = flow.get("device_code", "")

        # 步骤 3：保存提取出的结构化数据
        self._data_extractor.save_extracted_data(flow_data)

        # 步骤 4：统计当前同类型流程总数
        flow_counts = self._get_flow_counts_by_type()
        total_flows = flow_counts.get(flow_type, 0)
        logger.info(f"[flow_id={flow_id}] 数据提取完毕，开始 Qwen 大模型推理（总数={total_flows}）...")

        # 步骤 5：大模型流式推理（时钟感知直推）
        eval_local_sec = end_sec or start_sec
        has_playback_waited = False

        def wait_playback_reached() -> None:
            """在推流前等待前端画面播放到达流程结束时刻（附带 5.0 秒安全超时保护）."""
            nonlocal has_playback_waited
            if has_playback_waited or not self._get_playback_sec_fn:
                return
            target_sec = max(0.0, end_sec - 0.5)
            start_wait = time.time()
            while True:
                current_playback = self._get_playback_sec_fn()
                if current_playback >= target_sec:
                    break
                if time.time() - start_wait > 5.0:
                    logger.warning(
                        f"wait_playback_reached 等待前端播放进度超时(5.0s)，目标={target_sec:.2f}s, 当前={current_playback:.2f}s，强制放行"
                    )
                    break
                time.sleep(0.1)
            has_playback_waited = True

        def stream_callback(text_chunk: str) -> None:
            """逐 token 流式回调."""
            wait_playback_reached()
            if self._direct_fn:
                self._direct_fn("segment_report_stream", {
                    "localSec": eval_local_sec,
                    "tag": "segment_report_stream",
                    "data": {
                        "flow_id": flow_id,
                        "chunk": text_chunk,
                    },
                })

        try:
            qwen_report = self._qwen_evaluator.evaluate(
                flow_data,
                stream_callback=stream_callback,
                total_flows=total_flows,
            )
            report = {
                "flow_id": flow_id,
                "flow_type": qwen_report.get("flow_type", flow_type),
                "score": qwen_report.get("score", 0),
                "report_text": qwen_report.get("report_text", ""),
                "prompt": qwen_report.get("prompt", ""),
            }
        except Exception as eval_error:
            logger.error(f"[flow_id={flow_id}] Qwen 评估失败: {eval_error}", exc_info=True)
            report = {
                "flow_id": flow_id,
                "flow_type": flow_type,
                "score": 0,
                "report_text": f"评估失败: {eval_error}",
                "prompt": "",
            }

        # 步骤 6：保存单独 flow 的 LLM 评估结果
        self._save_flow_llm_response(flow_id, report, flow_data)

        with self._lock:
            self._segment_reports.append(report)
        self._save_segment_reports()

        # 步骤 7：推送最终完成报告
        wait_playback_reached()
        if self._direct_fn:
            self._direct_fn("segment_report", {
                "localSec": eval_local_sec,
                "tag": "segment_report",
                "data": {
                    "flow_id": flow_id,
                    "flow_type": report.get("flow_type"),
                    "score": report.get("score", 0),
                    "report_text": report.get("report_text", ""),
                    "flow_continue_sec": flow_data.get("flow_continue_sec"),
                },
            })

    def _get_flow_counts_by_type(self) -> Dict[str, int]:
        """统计各流程类型的数量（最大 flow_id）.

        Returns:
            Dict[str, int]: 各流程类型的计数映射.
        """
        eval_dir = os.path.join(self._result_dir, "evaluation")
        counts: Dict[str, int] = {}
        if not os.path.exists(eval_dir):
            return counts
        for filename in os.listdir(eval_dir):
            if filename.startswith("extracted_") and filename.endswith(".json"):
                name_part = filename[len("extracted_"):-len(".json")]
                last_underscore_idx = name_part.rfind("_")
                if last_underscore_idx <= 0:
                    continue
                flow_type = name_part[:last_underscore_idx]
                try:
                    current_fid = int(name_part[last_underscore_idx + 1:])
                    if current_fid > counts.get(flow_type, 0):
                        counts[flow_type] = current_fid
                except ValueError:
                    continue
        return counts

    def _save_flow_llm_response(self, flow_id: int, report: Dict[str, Any], flow_data: Dict[str, Any]) -> None:
        """保存单个 flow 的大模型评估完整数据.

        Args:
            flow_id (int): 流程编号.
            report (Dict[str, Any]): 评估报告字典.
            flow_data (Dict[str, Any]): 提取的多模态事实数据.
        """
        try:
            flow_type = flow_data.get("flow_type", "unknown")
            output_data = {
                "flow_id": flow_id,
                "flow_type": flow_type,
                "flow_start_sec": flow_data.get("flow_start_sec"),
                "flow_end_sec": flow_data.get("flow_end_sec"),
                "flow_continue_sec": flow_data.get("flow_continue_sec"),
                "score": report.get("score", 0),
                "report_text": report.get("report_text", ""),
                "prompt": report.get("prompt", ""),
                "flow_data": flow_data,
            }
            output_filename = f"qwen_response_{flow_type}_{flow_id}.json"
            output_path = os.path.join(self._qwen_dir, output_filename)
            with open(output_path, "w", encoding="utf-8") as file_handle:
                json.dump(output_data, file_handle, ensure_ascii=False, indent=2)
            logger.info(f"单个 flow 评估结果已保存: {output_path}")
        except Exception as write_error:
            logger.error(f"保存 flow_id={flow_id} 评估结果失败: {write_error}", exc_info=True)

    def _save_segment_reports(self) -> None:
        """保存汇总评估报告到 Qwen_segment_reports.json."""
        summary_path = os.path.join(self._qwen_dir, "Qwen_segment_reports.json")
        try:
            with self._lock:
                reports_list = list(self._segment_reports)
            with open(summary_path, "w", encoding="utf-8") as file_handle:
                json.dump(reports_list, file_handle, ensure_ascii=False, indent=2)
            logger.info(f"汇总评估报告已保存，共 {len(reports_list)} 条")
        except Exception as write_error:
            logger.error(f"保存汇总评估报告失败: {write_error}", exc_info=True)

    def finalize(self) -> None:
        """流水线结束时等待所有后台流程评估完成并落盘汇总."""
        logger.info("FlowEvaluationManager 执行 finalize 收尾，等待所有后台评估任务完成...")
        with self._eval_lock:
            active_futures = list(self._eval_futures.values())
        if active_futures:
            concurrent.futures.wait(active_futures, timeout=120)
        self._save_segment_reports()
