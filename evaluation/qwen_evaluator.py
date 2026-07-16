"""
大模型评估模块

使用 Qwen2.5-1.5B-Instruct 进行流程评估。
"""
import os
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("evaluation.qwen")

# 设置离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class QwenEvaluator:
    """
    Qwen 大模型评估器

    使用 Qwen2.5-1.5B-Instruct 进行流程评估。
    """

    def __init__(self, model_path: str = None):
        """
        初始化 Qwen 评估器

        Args:
            model_path: 模型路径
        """
        self._model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """加载模型"""
        if self._model is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            logger.info(f"加载 Qwen 模型: {self._model_path}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=True,
                local_files_only=True,
            )

            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
            )

            logger.info("Qwen 模型加载完成")

        except Exception as e:
            logger.error(f"Qwen 模型加载失败: {e}", exc_info=True)
            raise

    def evaluate(self, flow_data: Dict, stream_callback=None) -> Dict:
        """
        评估流程

        Args:
            flow_data: 流程数据
            stream_callback: 流式输出回调函数

        Returns:
            评估结果
        """
        self._load_model()

        try:
            # 构建 prompt
            prompt = self._build_prompt(flow_data)

            # 生成评估
            logger.info("开始 Qwen 评估...")
            report_text = self._generate(prompt, stream_callback)

            # 解析评分
            score = self._extract_score(report_text)

            logger.info(f"Qwen 评估完成，评分: {score}")

            return {
                "flow_type": flow_data.get("flow_type", "未知"),
                "score": score,
                "report_text": report_text,
                "prompt": prompt,
            }

        except Exception as e:
            logger.error(f"Qwen 评估失败: {e}", exc_info=True)
            return {
                "flow_type": flow_data.get("flow_type", "未知"),
                "score": 0,
                "report_text": f"评估失败: {e}",
                "prompt": "",
            }

    def _build_prompt(self, flow_data: Dict) -> str:
        """
        构建评估 prompt

        Args:
            flow_data: 流程数据

        Returns:
            prompt 字符串
        """
        flow_type = flow_data.get("flow_type", "未知")

        if flow_type == "supervision":
            return self._build_supervision_prompt(flow_data)
        elif flow_type == "self_ticket":
            return self._build_self_ticket_prompt(flow_data)
        elif flow_type == "info_notice":
            return self._build_info_notice_prompt(flow_data)
        else:
            return f"评估流程类型: {flow_type}"

    def _build_info_notice_prompt(self, flow_data: Dict) -> str:
        """
        构建信息通报评估 prompt

        Args:
            flow_data: 流程数据

        Returns:
            prompt 字符串
        """
        checklist = flow_data.get("content_checklist", {})
        voice_events = flow_data.get("voice_events", [])
        tracker_events = flow_data.get("tracker_events", [])
        gaze_events = flow_data.get("gaze_events", [])

        parts = []
        parts.append("你是核电站三向沟通（信息通报）合规检测评审专家。请根据以下流程数据进行评估。")
        parts.append("")
        parts.append("## 流程信息")
        parts.append(f"- 流程类型: 信息通报")
        parts.append(f"- 开始时间: {flow_data.get('flow_start_sec', 0)}s")
        parts.append(f"- 结束时间: {flow_data.get('flow_end_sec', 0)}s")
        parts.append(f"- 持续时间: {flow_data.get('flow_continue_sec', 0)}s")
        parts.append(f"- 触发来源: {flow_data.get('start_source', '未知')}")
        parts.append("")
        parts.append("## 内容检查清单")
        parts.append(f"- (1) 举手并高声喊出“信息通报”或“信息通告”: {'✅' if checklist.get('raise_hand_and_shout') else '❌'}")
        parts.append(f"- (2) 其他成员停下手中工作接受信息: {'✅' if checklist.get('others_stopped_and_listened') else '❌'}")
        parts.append(f"- (3) 确认团队成员均予以关注: {'✅' if checklist.get('others_attended') else '❌'}")
        parts.append(f"- (4) 发起者喊出“通报完毕”结束: {'✅' if checklist.get('shout_finished') else '❌'}")
        parts.append(f"- (5) 收到“收到”等语音给予回应: {'✅' if checklist.get('received_acknowledged') else '❌'}")
        parts.append("")
        parts.append("## 语音事件")
        for event in voice_events:
            parts.append(f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}")
        parts.append("")
        parts.append("## 跟踪与关注度事件")
        for event in tracker_events:
            parts.append(f"- [{event.get('localSec', 0)}s] {event.get('event', '')} ({event.get('state', '')})")
        parts.append("")
        parts.append("## 注视告警事件")
        for event in gaze_events:
            parts.append(f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}")
        parts.append("")
        parts.append("请根据以上数据，给出评分（满分10分）和评估报告。")

        return "\n".join(parts)

    def _build_supervision_prompt(self, flow_data: Dict) -> str:
        """
        构建监护制评估 prompt

        Args:
            flow_data: 流程数据

        Returns:
            prompt 字符串
        """
        checklist = flow_data.get("content_checklist", {})
        voice_events = flow_data.get("voice_events", [])
        tracker_events = flow_data.get("tracker_events", [])
        gaze_events = flow_data.get("gaze_events", [])

        parts = []
        parts.append("你是核电站监护制合规检测评审专家。请根据以下流程数据进行评估。")
        parts.append("")
        parts.append("## 流程信息")
        parts.append(f"- 流程类型: 监护制")
        parts.append(f"- 开始时间: {flow_data.get('flow_start_sec', 0)}s")
        parts.append(f"- 结束时间: {flow_data.get('flow_end_sec', 0)}s")
        parts.append(f"- 持续时间: {flow_data.get('flow_continue_sec', 0)}s")
        parts.append(f"- 触发来源: {flow_data.get('start_source', '未知')}")
        parts.append("")
        parts.append("## 内容检查清单")
        parts.append(f"- 九字码复述: {'✅' if checklist.get('code_repeat') else '❌'}")
        parts.append(f"- 执行操作: {'✅' if checklist.get('execution') else '❌'}")
        parts.append(f"- 核对确认: {'✅' if checklist.get('verification') else '❌'}")
        parts.append("")
        parts.append("## 语音事件")
        for event in voice_events:
            parts.append(f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}")
        parts.append("")
        parts.append("## 跟踪事件")
        for event in tracker_events:
            parts.append(f"- [{event.get('localSec', 0)}s] {event.get('event', '')} ({event.get('state', '')})")
        parts.append("")
        parts.append("## 注视告警事件")
        for event in gaze_events:
            parts.append(f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}")
        parts.append("")
        parts.append("请根据以上数据，给出评分（满分10分）和评估报告。")

        return "\n".join(parts)

    def _build_self_ticket_prompt(self, flow_data: Dict) -> str:
        """
        构建自唱票评估 prompt

        Args:
            flow_data: 流程数据

        Returns:
            prompt 字符串
        """
        parts = []
        parts.append("你是核电站自唱票合规检测评审专家。请根据以下流程数据进行评估。")
        parts.append("")
        parts.append("## 流程信息")
        parts.append(f"- 流程类型: 自唱票")
        parts.append(f"- 开始时间: {flow_data.get('flow_start_sec', 0)}s")
        parts.append(f"- 结束时间: {flow_data.get('flow_end_sec', 0)}s")
        parts.append(f"- 持续时间: {flow_data.get('flow_continue_sec', 0)}s")
        parts.append(f"- 设备代码: {flow_data.get('device_code', '未知')}")
        parts.append("")
        parts.append("## 内容检查清单")
        parts.append(f"- 九字码读出: {'✅' if flow_data.get('code_read') else '❌'}")
        parts.append(f"- 操作执行: {'✅' if flow_data.get('operation_executed') else '❌'}")
        parts.append(f"- 确认闭环: {'✅' if flow_data.get('confirm_closed') else '❌'}")
        parts.append("")
        parts.append("请根据以上数据，给出评分（满分10分）和评估报告。")

        return "\n".join(parts)

    def _generate(self, prompt: str, stream_callback=None) -> str:
        """
        生成评估报告

        Args:
            prompt: 输入 prompt
            stream_callback: 流式输出回调函数

        Returns:
            评估报告文本
        """
        try:
            from transformers import TextIteratorStreamer

            # 准备输入
            messages = [{"role": "user", "content": prompt}]
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self._model.device)

            # 生成参数
            generation_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": 2048,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            }

            # 流式输出
            if stream_callback:
                streamer = TextIteratorStreamer(
                    self._tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                generation_kwargs["streamer"] = streamer

                # 在单独线程中生成
                thread = threading.Thread(
                    target=self._model.generate,
                    kwargs=generation_kwargs,
                )
                thread.start()

                # 收集输出
                report_text = ""
                for text in streamer:
                    report_text += text
                    stream_callback(text)

                thread.join()
                return report_text

            else:
                # 非流式输出
                outputs = self._model.generate(**generation_kwargs)
                report_text = self._tokenizer.decode(
                    outputs[0][input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                return report_text

        except Exception as e:
            logger.error(f"生成失败: {e}", exc_info=True)
            raise

    def _extract_score(self, report_text: str) -> int:
        """
        从报告文本中提取评分

        Args:
            report_text: 报告文本

        Returns:
            评分
        """
        import re

        # 尝试匹配 "总分：X" 或 "得分：X" 或 "X/10"
        patterns = [
            r"总分[：:]\s*(\d+)",
            r"得分[：:]\s*(\d+)",
            r"(\d+)/10",
            r"评分[：:]\s*(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, report_text)
            if match:
                return int(match.group(1))

        # 默认返回 5 分
        logger.warning("无法从报告中提取评分，默认返回 5 分")
        return 5
