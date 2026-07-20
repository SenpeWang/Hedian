"""
流程事件记录器

职责：
- 订阅事件流中的 FLOW_STARTED / FLOW_ENDED
- 维护活跃流程和已完成流程列表
- 将流程开始/结束时间、类型、来源等信息保存到 result_dir/rules/flow_events.json

规则层（rules）负责流程识别，因此流程记录也归属 rules 层。
"""
import os
import json
import threading
import logging
from typing import Dict, List, Optional

from core.event_bus import EventBus, EventTopic

logger = logging.getLogger("rules.flow_recorder")


class FlowEventRecorder:
    """流程事件记录器 — 保存每个流程的开始/结束时间到 rules/flow_events.json"""

    def __init__(self, event_bus: EventBus):
        """
        初始化流程事件记录器

        Args:
            event_bus: 事件总线实例
        """
        self._event_bus = event_bus
        self._result_dir: Optional[str] = None
        self._active_flows: Dict[int, dict] = {}
        self._completed_flows: List[dict] = []
        self._lock = threading.Lock()

        # 订阅流程事件
        self._event_bus.subscribe(EventTopic.FLOW_STARTED, self._on_flow_started)
        self._event_bus.subscribe(EventTopic.FLOW_ENDED, self._on_flow_ended)
        logger.info("FlowEventRecorder 已订阅 FLOW_STARTED/FLOW_ENDED")

    def set_result_dir(self, result_dir: str) -> None:
        """设置结果目录（由主流程在每次推理启动时调用）"""
        with self._lock:
            self._result_dir = result_dir
            self._active_flows.clear()
            self._completed_flows.clear()
        logger.info(f"FlowEventRecorder 结果目录设置为: {result_dir}")

    def _on_flow_started(self, msg: dict) -> None:
        """处理流程开始事件"""
        data = msg.get("data", {})
        flow_id = data.get("flow_id")
        if flow_id is None:
            logger.warning("FLOW_STARTED 事件缺少 flow_id")
            return

        with self._lock:
            self._active_flows[flow_id] = dict(data)
        logger.info(f"记录流程开始 flow_id={flow_id} type={data.get('flow_type')}")

    def _on_flow_ended(self, msg: dict) -> None:
        """处理流程结束事件：合并开始/结束信息并保存"""
        data = msg.get("data", {})
        flow_id = data.get("flow_id")
        if flow_id is None:
            logger.warning("FLOW_ENDED 事件缺少 flow_id")
            return

        with self._lock:
            start_data = self._active_flows.pop(flow_id, {})
            # 以 FLOW_ENDED 数据为准，合并开始阶段的信息
            flow_record = {
                "flow_id": flow_id,
                "flow_type": data.get("flow_type", start_data.get("flow_type", "unknown")),
                "flow_start_sec": start_data.get("flow_start_sec", data.get("flow_start_sec", 0)),
                "start_source": start_data.get("start_source", data.get("start_source", "unknown")),
                "flow_end_sec": data.get("flow_end_sec", 0),
                "end_source": data.get("end_source", "unknown"),
                "flow_continue_sec": data.get(
                    "flow_continue_sec",
                    round(data.get("flow_end_sec", 0) - start_data.get("flow_start_sec", 0), 2),
                ),
                "device_code": data.get("device_code", start_data.get("device_code", "")),
                "content_checklist": data.get("content_checklist", start_data.get("content_checklist", {})),
                "sub_flows": data.get("sub_flows", start_data.get("sub_flows", [])),
            }
            self._completed_flows.append(flow_record)

        logger.info(f"记录流程结束 flow_id={flow_id} type={flow_record['flow_type']}")
        self._save_flow_events()

    def _save_flow_events(self) -> None:
        """保存流程事件到 result_dir/rules/flow_events.json"""
        with self._lock:
            result_dir = self._result_dir
            flows = list(self._completed_flows)

        if not result_dir:
            logger.warning("结果目录未设置，跳过保存 flow_events.json")
            return

        try:
            rules_dir = os.path.join(result_dir, "rules")
            os.makedirs(rules_dir, exist_ok=True)
            path = os.path.join(rules_dir, "flow_events.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(flows, f, ensure_ascii=False, indent=2)
            logger.info(f"保存 {len(flows)} 个流程事件到 {path}")
        except Exception as e:
            logger.error(f"保存流程事件失败: {e}", exc_info=True)

    def get_completed_flows(self) -> List[dict]:
        """获取已完成的流程列表（只读副本）"""
        with self._lock:
            return list(self._completed_flows)
