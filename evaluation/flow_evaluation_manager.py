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
import time
import logging
import threading
import concurrent.futures
import redis
from typing import Dict, List, Callable

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

        # 默认模型路径（Qwen3-8B，部署在 GPU 0）
        if model_path is None:
            from pathlib import Path
            model_path = str(Path(__file__).parent.parent / "models" / "evaluation" / "Qwen3-8B")

        # 数据提取器
        redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        self._data_extractor = FlowDataExtractor(result_dir, redis_client=redis_client)

        # 评估器
        self._qwen_evaluator = QwenEvaluator(model_path=model_path)

        # 流程管理
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

    def set_result_dir(self, result_dir: str) -> None:
        """动态更新评估结果目录"""
        self._result_dir = result_dir
        self._qwen_dir = os.path.join(result_dir, "qwen")
        os.makedirs(self._qwen_dir, exist_ok=True)
        if hasattr(self, "_data_extractor") and self._data_extractor:
            self._data_extractor._result_dir = result_dir
        logger.info(f"FlowEvaluationManager 结果目录动态更新为: {result_dir}")

    def _on_flow_started(self, msg: dict) -> None:
        """处理流程开始事件：仅用于实时将流程启动信号推送到前端显示，不保存冗余状态"""
        data = msg.get("data", {})
        flow_id = data.get("flow_id", 0)
        flow_type = data.get("flow_type", "unknown")
        ts = data.get("flow_start_sec", msg.get("ts", 0))

        if self._display_fn:
            self._display_fn("flow_start", {
                "localSec": ts,
                "tag": "flow_start",
                "data": {
                    "flow_id": flow_id,
                    "flow_type": flow_type,
                    "flow_start_sec": ts,
                    "start_source": data.get("start_source", "unknown"),
                },
            })

        logger.info(f"流程开始 flow_id={flow_id} type={flow_type} @{ts:.1f}s")

    def _on_flow_ended(self, msg: dict) -> None:
        """处理流程结束事件：直接利用消息中携带的完备 flow 数据，无需从内存中 pop 合并"""
        flow = msg.get("data", {})
        flow_id = flow.get("flow_id", 0)
        if not flow_id:
            logger.warning("FLOW_ENDED 事件缺少 flow_id")
            return

        with self._lock:
            self._completed_flows.append(flow)

        if self._display_fn:
            self._display_fn("flow_end", {
                "localSec": flow.get("flow_end_sec", 0),
                "tag": "flow_end",
                "data": flow,
            })

        logger.info(f"流程结束 flow_id={flow_id}")

        future = self._executor.submit(self._extract_save_and_evaluate, flow)
        with self._eval_lock:
            self._eval_futures[f"extract_{flow_id}"] = future
        future.add_done_callback(
            lambda f, fid=flow_id: self._on_extract_done(fid)
        )

    def _on_extract_done(self, flow_id: int) -> None:
        """数据提取完成回调"""
        with self._eval_lock:
            self._eval_futures.pop(f"extract_{flow_id}", None)

    def _extract_save_and_evaluate(self, flow: Dict) -> None:
        """提取流程数据 → 保存到 extracted_{flow_type}_{flow_id}.json → 立即评估"""
        start_sec = flow.get("flow_start_sec", 0)
        end_sec = flow.get("flow_end_sec") or start_sec

        self._data_extractor._wait_all_modules(end_sec, timeout=300)

        self._event_bus.publish(EventTopic.SAVE_KEY_MOMENTS, {"flow_id": flow.get("flow_id")}, ts=end_sec)
        time.sleep(2.0)

        voice_events, tracker_events, gaze_events, behavior_events = self._data_extractor.extract(
            start_sec, end_sec, wait=False, timeout=300
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
            "behavior_events": behavior_events,
        }

        if flow.get("flow_type") in ("supervision", "info_notice"):
            flow_data["content_checklist"] = flow.get("content_checklist", {})
        if flow.get("flow_type") == "self_ticket":
            flow_data["device_code"] = flow.get("device_code", "")

        self._data_extractor.save_extracted_data(flow_data)
        logger.info(f"flow_id={flow.get('flow_id')} 数据已提取保存，准备立即评估")

        flow_id = flow.get("flow_id")
        flow_type = flow.get("flow_type", "unknown")
        saved_data = self._load_extracted_flow(flow_type, flow_id)
        if not saved_data:
            logger.error(
                f"flow_type={flow_type} flow_id={flow_id} 读取 "
                f"extracted_{flow_type}_{flow_id}.json 失败，跳过评估"
            )
            return

        flow_counts = self._get_flow_counts_by_type()
        total_flows = flow_counts.get(flow_type, 0)
        logger.info(
            f"flow_type={flow_type} flow_id={flow_id} 开始评估，"
            f"当前 {flow_type} 类型共 {total_flows} 个流程，"
            f"全部流程计数: {flow_counts}"
        )

        future = self._executor.submit(self._do_evaluate, saved_data, total_flows)
        with self._eval_lock:
            self._eval_futures[flow_id] = future
        future.add_done_callback(
            lambda f, fid=flow_id: self._on_eval_done(f, fid)
        )

    def _load_extracted_flow(self, flow_type: str, flow_id: int) -> Dict:
        """从 extracted_{flow_type}_{flow_id}.json 读取流程数据"""
        path = os.path.join(
            self._result_dir, "evaluation", f"extracted_{flow_type}_{flow_id}.json"
        )
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取 extracted_{flow_type}_{flow_id}.json 失败: {e}")
            return {}

    def _get_flow_counts_by_type(self) -> Dict[str, int]:
        """按 flow_type 统计每种流程类型的数量（max flow_id）"""
        eval_dir = os.path.join(self._result_dir, "evaluation")
        counts: Dict[str, int] = {}
        if not os.path.exists(eval_dir):
            return counts
        for fn in os.listdir(eval_dir):
            if fn.startswith("extracted_") and fn.endswith(".json"):
                name_part = fn[len("extracted_"):-len(".json")]
                # flow_type 可能含下划线（如 self_ticket），从右边分
                idx = name_part.rfind("_")
                if idx <= 0:
                    continue
                flow_type = name_part[:idx]
                try:
                    fid = int(name_part[idx + 1:])
                    if fid > counts.get(flow_type, 0):
                        counts[flow_type] = fid
                except ValueError:
                    continue
        return counts

    def _do_evaluate(self, flow_data: Dict, total_flows: int = 0) -> tuple:
        """执行评估，返回 (评估结果, 流程数据)"""
        flow_id = flow_data["flow_id"]
        eval_local_sec = flow_data.get("flow_end_sec", flow_data.get("flow_start_sec", 0))

        def stream_cb(text_chunk):
            if self._display_fn:
                self._display_fn("segment_report_stream", {
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
                stream_callback=stream_cb,
                total_flows=total_flows,
            )

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
                "prompt": "",
            }
            flow_data = {"flow_id": flow_id, "flow_continue_sec": 0}

        # 给 report 加上 flow_id 便于追溯
        report["flow_id"] = flow_id

        # 单独保存该 flow 的 LLM 评估结果（包含 prompt + 报告 + 流程数据）
        self._save_flow_llm_response(flow_id, report, flow_data)

        with self._lock:
            self._segment_reports.append(report)

        self._save_segment_reports()

        if self._display_fn:
            self._display_fn("segment_report", {
                "localSec": flow_data.get(
                    "flow_end_sec",
                    flow_data.get("flow_start_sec", 0),
                ),
                "tag": "segment_report",
                "data": {
                    "flow_id": flow_id,
                    "flow_type": report.get("flow_type"),
                    "score": report.get("score", 0),
                    "report_text": report.get("report_text", ""),
                    "flow_continue_sec": flow_data.get("flow_continue_sec"),
                },
            })

    def _save_flow_llm_response(self, flow_id: int, report: Dict, flow_data: Dict) -> None:
        """保存单个 flow 的 LLM 评估结果到 qwen_response_{flow_id}.json"""
        try:
            payload = {
                "flow_id": flow_id,
                "flow_type": report.get("flow_type", "未知"),
                "flow_start_sec": flow_data.get("flow_start_sec"),
                "flow_end_sec": flow_data.get("flow_end_sec"),
                "flow_continue_sec": flow_data.get("flow_continue_sec"),
                "score": report.get("score", 0),
                "prompt": report.get("prompt", ""),
                "report_text": report.get("report_text", ""),
                "flow_data": flow_data,
            }
            path = os.path.join(self._qwen_dir, f"qwen_response_{flow_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(f"flow_id={flow_id} LLM 响应已保存到 {path}")
        except Exception as e:
            logger.error(f"保存 flow_id={flow_id} LLM 响应失败: {e}", exc_info=True)

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
        """视频结束时的清理：等待所有评估任务完成"""
        for flow in self._completed_flows:
            fid = flow.get("flow_id")
            ftype = flow.get("flow_type", "unknown")
            already = any(r.get("flow_id") == fid for r in self._segment_reports)
            extracted_path = os.path.join(
                self._result_dir, "evaluation", f"extracted_{ftype}_{fid}.json"
            )
            if not already and not os.path.exists(extracted_path):
                self._extract_save_and_evaluate(flow)

        # 阻塞等待所有大模型数据提取与评估任务全部完成
        logger.info("开始等待所有大模型数据提取与评估任务完成...")
        import time
        start_wait = time.time()
        while True:
            with self._eval_lock:
                active_futures = list(self._eval_futures.values())
            if not active_futures:
                break
            
            # 等待当前的所有活跃任务
            concurrent.futures.wait(active_futures, timeout=1.0)
            
            # 超时保护（300秒）
            if time.time() - start_wait > 300.0:
                logger.warning("等待大模型评估完成超时！强行终止")
                break
            time.sleep(0.5)
        logger.info("所有大模型数据提取与评估任务已全部完成")
