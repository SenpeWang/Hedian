"""
意图分类模块

从语音转录结果中分类意图，发布到消息总线。
"""
import re
import logging
from typing import List, Dict, Optional

from core.message_bus import MessageBus, MsgType

logger = logging.getLogger("module.voice.intent")

# 意图分类关键词
DEVICE_PATTERN = re.compile(r"(1ES\w+|T1RPA\w+|LCO\w+|RPA\w+|SM3)")
ACTION_VERBS = ["开启", "关闭", "长按", "调出", "停运"]
CONFIRM_WORDS = ["好", "确认", "没问题", "收到", "明白", "正确"]
VERIFY_WORDS = ["核对", "核实", "验证", "检查"]

# 意图 -> 关键时刻映射
INTENT_TO_KEY_MOMENT = {
    "监护请求": "监护",
    "监护确认": "确认",
    "操作指令": "执行",
    "操作说明": "执行",
    "核对确认": "核对",
    "确认": "好",
    "实验结束": "结束",
    "操作结束": "结束",
}

KEY_MOMENT_TO_TYPE = {
    "监护": "信号信号",
    "结束": "信号信号",
    "执行": "执行信号",
    "确认": "核对信号",
    "好": "核对信号",
    "核对": "核对信号",
    "九字码": "设备信号",
}


def classify_intent(text: str, prev_intent: str = "") -> str:
    """
    基于关键词规则的意图分类

    Args:
        text: 输入文本
        prev_intent: 前一个意图

    Returns:
        意图标签
    """
    if "请求监护" in text or "请监护" in text or "申请监护" in text:
        return "监护请求"

    if prev_intent == "监护请求" and any(w in text for w in ["好", "收到", "明白"]) and len(text) <= 6:
        return "监护确认"

    if "实验结束" in text or "试验结束" in text or "实验完成" in text or "试验完成" in text:
        return "实验结束"

    if "框架退出" in text:
        return "操作结束"

    has_device = bool(DEVICE_PATTERN.search(text))
    has_action = any(v in text for v in ACTION_VERBS)

    if has_device and has_action:
        return "操作指令"

    if any(v in text for v in VERIFY_WORDS):
        return "核对确认"

    stripped = text.replace("好,", "").replace("好，", "").strip()
    if stripped in CONFIRM_WORDS or text.strip() in CONFIRM_WORDS:
        return "确认"

    if has_device:
        return "操作说明"

    return "其他"


class IntentClassifier:
    """
    意图分类器

    接收语音转录的词列表，分句、纠错、分类意图，
    然后将事件发布到消息总线。
    """

    def __init__(self, bus: MessageBus, config: dict = None):
        """
        初始化意图分类器

        Args:
            bus: 消息总线
            config: 配置字典
        """
        self._bus = bus
        self._config = config or {}
        self._sentence_gap_sec = self._config.get("sentence_gap_sec", 1.5)
        self._last_device_code = ""
        self._last_device_time = 0.0
        self._prev_intent = ""
        self._last_intent_time = 0.0

    def classify(self, words: List[Dict]) -> List[Dict]:
        """
        处理语音转录的词列表

        Args:
            words: 词列表 [{"word": str, "start": float, "end": float}, ...]

        Returns:
            事件列表
        """
        if not words:
            return []

        words = [w for w in words if w is not None and w.get("word")]
        if not words:
            return []

        # 时间衰减：如果距离上次意图超过30秒，清空 prev_intent
        if words and self._last_intent_time > 0:
            gap = words[0]["start"] - self._last_intent_time
            if gap > 30.0:
                self._prev_intent = ""
        self._last_intent_time = 0.0

        events = []

        # 分句：按停顿切分
        sentences = []
        cur = []
        for w in words:
            if cur and w["start"] - cur[-1]["end"] > self._sentence_gap_sec:
                sentences.append(cur)
                cur = [w]
            else:
                cur.append(w)
        if cur:
            sentences.append(cur)

        for sent_words in sentences:
            text = "".join(w["word"] for w in sent_words)
            if not text.strip():
                continue

            audio_ts = round(sent_words[0]["start"], 2)

            # 意图分类
            intent = classify_intent(text, self._prev_intent)
            self._prev_intent = intent
            self._last_intent_time = audio_ts

            # 设备码提取
            m = DEVICE_PATTERN.search(text)
            device = m.group(1) if m else ""

            # 关键时刻
            km = INTENT_TO_KEY_MOMENT.get(intent)
            if km is None and device:
                km = "九字码"
            km_type = KEY_MOMENT_TO_TYPE.get(km, "") if km else ""

            # 构建事件
            ev = {
                "localSec": audio_ts,
                "text": text,
                "intent": intent,
                "device": device,
                "key_moment": km or "",
                "km_type": km_type,
            }
            events.append(ev)

            # 发布到 bus
            self._bus.publish(MsgType.VOICE_INTENT, ev, ts=audio_ts)

            # 设备码事件
            if device:
                self._bus.publish(MsgType.VOICE_DEVICE_CODE, {
                    "localSec": audio_ts,
                    "device": device,
                    "text": text,
                    "intent": intent,
                }, ts=audio_ts)

                # 设备码重复检测
                if device == self._last_device_code and audio_ts - self._last_device_time < 60:
                    self._bus.publish(MsgType.VOICE_DEVICE_CODE, {
                        "localSec": audio_ts,
                        "device": device,
                        "text": text,
                        "intent": intent,
                        "repeat": True,
                    }, ts=audio_ts)
                self._last_device_code = device
                self._last_device_time = audio_ts

        return events

    def reset(self):
        """重置状态"""
        self._prev_intent = ""
        self._last_intent_time = 0.0
        self._last_device_code = ""
        self._last_device_time = 0.0
