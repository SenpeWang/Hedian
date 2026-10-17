"""语音模块入口：流式转录音频、逐段成句并提取关键事件推送.

继承 BaseModule，实现统一接口；转录产出的词表经关键词与设备码匹配后，
既推送推理流供前端展示，也推送事件流供规则层判定，并实时增量落盘。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.base_module import BaseModule
from core.event_bus import EventBus, EventTopic
from core.inference_stream import InferenceStream
from core.path_manager import PathConfig

from modules.voice.sentence_assembler import SentenceAssembler
from modules.voice.speech_transcriber import (
    SpeechTranscriber,
    normalize_devices_in_text,
    process_transcribed_words,
)
from modules.voice.storage_voice import VoiceStorage
from modules.voice.streaming_source import is_live_source

logger = logging.getLogger("module.voice")


class VoiceModule(BaseModule):
    """语音模块.

    负责语音转文字和意图分类。
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: Dict[str, Any],
        paths: PathConfig,
        inference_stream: InferenceStream,
    ) -> None:
        """初始化语音模块状态.

        Args:
            event_bus: 跨进程消息总线.
            config: 配置字典.
            paths: 路径配置.
            inference_stream: 推理流写入端.
        """
        super().__init__(event_bus, config, paths, inference_stream)
        self._transcriber: Optional[SpeechTranscriber] = None
        self._result_storage: Optional[VoiceStorage] = None
        self._events: List[Dict[str, Any]] = []

    @property
    def module_name(self) -> str:
        """模块名称."""
        return "voice"

    def initialize(self) -> bool:
        """初始化语音模块."""
        try:
            # 初始化语音转文字器
            voice_config = self.config.get("voice", {})
            self._transcriber = SpeechTranscriber(
                model_path=voice_config.get("model_path"),
                aligner_path=voice_config.get("aligner_path"),
                sample_rate=voice_config.get("sample_rate", 16000),
                torch_dtype=voice_config.get("torch_dtype"),
                vad_noise_db=voice_config.get("vad_noise_db", -35.0),
                vad_min_silence_sec=voice_config.get(
                    "vad_min_silence_sec", 0.8
                ),
                vad_speech_pad_sec=voice_config.get("vad_speech_pad_sec", 0.2),
                vad_min_speech_sec=voice_config.get(
                    "vad_min_speech_sec", 0.35
                ),
                vad_max_chunk_sec=voice_config.get("vad_max_chunk_sec", 30.0),
            )

            # 初始化结果存储
            self._result_storage = VoiceStorage(self.paths)

            logger.info("语音模块初始化完成")
            return True

        except Exception as init_error:
            logger.error(f"语音模块初始化失败: {init_error}", exc_info=True)
            return False

    def process_video(self, video_path: str) -> None:
        """处理视频/音频源，逐段转录、逐段分类、逐段推送.

        走流式流水线：ffmpeg 管道增量读音频 + 在线切片 + 逐段解码 + 跨段成句，
        段闭合即推送（原批处理链路已移除）。

        Args:
            video_path: 视频/音频源路径（可为直播 URL）.
        """
        try:
            self._process_video_stream(video_path)
        except Exception as process_error:
            logger.error("语音处理失败: %s", process_error, exc_info=True)

    def _process_video_stream(self, video_path: str) -> None:
        """流式流水线：ffmpeg 管道增量读音频 + 在线切片 + 逐段解码 + 跨段成句.

        不落盘 `audio.wav`、不做整文件静音扫描、不切临时 wav；音频边读边切，
        段一闭合立即解码并推送，因此首个结果不需要等待整段音频就绪，内存占用
        也与音频总时长无关。

        Args:
            video_path: 视频/音频源路径（可为直播 URL）.
        """
        voice_config = self.config.get("voice", {})
        source = self._resolve_source(video_path)
        logger.info("开始流式语音转文字: %s", source)

        # 保护词：不可被句边界切开的词（岗位词、复合关键词）
        #   null（或键缺失）→ 自动从 KEYWORD_LABELS 派生
        #   []            → 关闭保护
        #   [词1, 词2]     → 自定义词表
        protected_words = voice_config.get("protected_words")

        # 跨段全局词流成句：即使切片点落在词中间（如 "收"|"到"、"二"|"回路"），
        # 也能在成句阶段拼回；保护词内部一律不允许断句
        assembler = SentenceAssembler(
            sentence_gap_sec=voice_config.get("sentence_gap_sec", 1.0),
            max_sentence_sec=voice_config.get("max_sentence_sec", 30.0),
            protected_words=protected_words,
            protected_hold_sec=voice_config.get("protected_hold_sec", 2.0),
        )

        def handle_sentence(sentence_words: List[Dict[str, Any]]) -> None:
            """把一句（可能由多段拼成）转成事件并推送、落盘.

            Args:
                sentence_words: 已封口的句子词表.
            """
            events = process_transcribed_words(
                sentence_words,
                sentence_gap_sec=voice_config.get("sentence_gap_sec", 1.0),
                protected_words=protected_words,
            )
            if not events:
                return
            for event in events:
                local_sec = event.get("localSec", 0.0)
                text = event.get("text", "")
                key_moment = event.get("key_moment", "")

                # 推理流：实时转录结果（带关键词 keys 用于前端高亮红字展示）
                if text:
                    display_text = normalize_devices_in_text(text)
                    keys = event.get("keys", [])
                    if not keys and key_moment:
                        keys = [key_moment]
                    self.push_display(
                        "voice",
                        {
                            "localSec": local_sec,
                            "tag": "text",
                            "data": {"text": display_text, "keys": keys},
                        },
                    )
                    self.inference_stream.update_module_snapshot(
                        "voice", {"latest_text": display_text}
                    )

                # key_moment 既推推理流又推事件流
                if key_moment:
                    self.push_display(
                        "voice",
                        {
                            "localSec": local_sec,
                            "tag": "key_moment",
                            "data": {"key_moment": key_moment},
                        },
                    )
                    self.push_event(
                        EventTopic.VOICE_KEY_MOMENT,
                        {
                            "localSec": local_sec,
                            "key_moment": key_moment,
                            # 同句全部关键词（已按句中出现位置排序）：供规则层判断
                            # "请讲"之后是否紧跟"收到"这类同句信号，避免被漏掉
                            "keys": event.get("keys", []),
                        },
                        timestamp=local_sec,
                    )

                # 保存归一化后的文本，确保 voice_full_text.json 中的设备码为纯英文数字
                event["text"] = display_text if text else event.get("text", "")
                self._events.append(event)

            # 实时增量落盘：每句产出事件后立即原子写盘，确保 evaluation 随时可提取
            if self._result_storage:
                self._result_storage.save_results(self._run_id, self._events)

        def on_segment(
            words: List[Dict[str, Any]],
            seg_start: float,
            seg_end: float,
            total_duration: float,
        ) -> None:
            """每段解码完成后的回调：成句并推进进度.

            Args:
                words: 该段带时间戳的词表.
                seg_start: 段起始（秒）.
                seg_end: 段结束（秒）.
                total_duration: 总时长（秒）；直播源为 0.
            """
            if not words:
                return
            for sentence_words in assembler.push(words):
                handle_sentence(sentence_words)
            # 句尾停顿已被确认的句子立即交付，不必等到下一个词到达
            for sentence_words in assembler.tick(seg_end):
                handle_sentence(sentence_words)
            self.update_progress(seg_end, total_duration)

        def on_progress(current_sec: float, total_sec: float) -> None:
            """音频读取进度回调：仅用于按游标及时封句，不改动对齐进度.

            对齐进度（per-source localSec）必须反映**已解码**的位置，而不是
            **已读取**的位置：ffmpeg 会以远快于实时的速度预读，若用读取游标写入
            进度，`InferenceSync` 的 global_sec 会被推快，导致语音文字落后于
            其他模态的显示节奏。

            Args:
                current_sec: 已读取到的绝对时间（秒）.
                total_sec: 总时长（秒）.
            """
            for sentence_words in assembler.tick(current_sec):
                handle_sentence(sentence_words)

        self._transcriber.transcribe_stream(
            source=source,
            progress_callback=on_progress,
            on_segment=on_segment,
            segmenter_name=voice_config.get("segmenter", "silence_vad"),
            segmenter_config={
                # silence_vad 策略参数
                "noise_db": voice_config.get("vad_noise_db", -35.0),
                "min_silence": voice_config.get("vad_min_silence_sec", 0.8),
                "speech_pad": voice_config.get("vad_speech_pad_sec", 0.2),
                "min_speech": voice_config.get("vad_min_speech_sec", 0.35),
                "max_chunk": voice_config.get("vad_max_chunk_sec", 30.0),
                "split_search_sec": voice_config.get(
                    "vad_split_search_sec", 1.0
                ),
                "split_margin_db": voice_config.get(
                    "vad_split_margin_db", 6.0
                ),
                # fixed_window 策略参数
                "window_sec": voice_config.get("fixed_window_sec", 20.0),
                "overlap_sec": voice_config.get(
                    "fixed_window_overlap_sec", 5.0
                ),
            },
            boundary_overlap_sec=voice_config.get("boundary_overlap_sec", 0.4),
            block_sec=voice_config.get("stream_block_sec", 0.2),
        )

        for sentence_words in assembler.flush():
            handle_sentence(sentence_words)

        logger.info(
            "流式语音处理完成，共 %d 个事件，句子组装器: %s",
            len(self._events),
            assembler.describe(),
        )

    def _resolve_source(self, video_path: str) -> str:
        """把输入源规范化为 ffmpeg 可直接读取的形式.

        直播 URL 原样返回；相对路径按项目根目录补全；空值回退到配置中的 front 视频。

        Args:
            video_path: 传入的源路径或 URL.

        Returns:
            可直接交给 ffmpeg 的源.
        """
        if video_path and is_live_source(video_path):
            return video_path
        if not video_path:
            return str(
                self.paths.base_dir
                / self.config.get("voice", {}).get(
                    "source",
                    self.config.get("videos", {}).get(
                        "front", "data/videos/camFRONT.mpg"
                    ),
                )
            )
        if (
            not Path(video_path).is_absolute()
            and not Path(video_path).exists()
        ):
            return str(self.paths.base_dir / video_path)
        return video_path

    def save_results(self, run_id: str) -> None:
        """保存语音结果（委托给 VoiceStorage）."""
        self._result_storage.save_results(run_id, self._events)
