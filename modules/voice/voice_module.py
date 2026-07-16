"""
语音模块入口

继承 BaseModule，实现统一接口。
"""
import os
import logging

from core.base_module import BaseModule
from core.event_bus import EventBus, EventTopic
from core.display_buffer import DisplayBuffer
from core.path_manager import PathConfig

from modules.voice.speech_transcriber import SpeechTranscriber, process_transcribed_words
from modules.voice.storage import VoiceResultStorage

logger = logging.getLogger("module.voice")


class VoiceModule(BaseModule):
    """
    语音模块

    负责语音转文字和意图分类。
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: dict,
        paths: PathConfig,
        display_buffer: DisplayBuffer,
    ):
        super().__init__(event_bus, config, paths, display_buffer)
        self._transcriber = None
        self._intent_classifier = None
        self._result_storage = None
        self._events = []

    @property
    def module_name(self) -> str:
        return "voice"

    def initialize(self) -> bool:
        """初始化语音模块"""
        try:
            # 初始化语音转文字器
            voice_config = self.config.get("voice", {})
            self._transcriber = SpeechTranscriber(
                model_path=voice_config.get("model_path"),
                sample_rate=voice_config.get("sample_rate", 16000),
            )

            # 初始化结果存储
            self._result_storage = VoiceResultStorage(self.paths)

            logger.info("语音模块初始化完成")
            return True

        except Exception as e:
            logger.error(f"语音模块初始化失败: {e}", exc_info=True)
            return False

    def process_video(self, video_path: str) -> None:
        """处理视频，逐段转录、逐段分类、逐段推送（与其他模块时间对齐）"""
        try:
            # 提取音频
            audio_path = self._extract_audio(video_path)
            if not audio_path:
                logger.error("音频提取失败")
                return

            # 逐段转录，每段完成后立即分类并推送
            logger.info("开始语音转文字（逐段模式）...")

            def on_segment_done(words, seg_start, seg_end, total_dur):
                """每段转录完成后的回调：分句后按需推送推理流与事件流"""
                if not words:
                    return
                events = process_transcribed_words(words, sentence_gap_sec=1.0)
                for event in events:
                    local_sec = event.get("localSec", 0.0)
                    text = event.get("text", "")
                    key_moment = event.get("key_moment", "")

                    # 1. 推送推理流：完全的实时转录结果 (只包含时间轴和转录文本)
                    if text:
                        self.push_display("voice", {
                            "localSec": local_sec,
                            "text": text
                        })

                    # 2. 推送事件流：仅核心关键字（含 localSec 和 key_moment）
                    if key_moment:
                        self.push_event(EventTopic.VOICE_KEY_MOMENT, {
                            "localSec": local_sec,
                            "key_moment": key_moment
                        }, ts=local_sec)

                    self._events.append(event)
                self.update_progress(seg_end, total_dur)
                logger.debug("voice segment %.1f-%.1fs: %d events", seg_start, seg_end, len(events))

            def voice_progress(current, total_duration):
                self.update_progress(current, total_duration)

            self._transcriber.transcribe(
                audio_path,
                progress_callback=voice_progress,
                on_segment=on_segment_done,
            )

            logger.info("语音处理完成，共 %d 个事件", len(self._events))

        except Exception as e:
            logger.error("语音处理失败: %s", e, exc_info=True)

    def _extract_audio(self, video_path: str) -> str:
        """从视频提取音频"""
        import subprocess

        audio_path = str(self.paths.get_result_path(
            run_id="temp",
            module="voice",
            filename="audio.wav",
        ))

        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-ar", "16000", "-ac", "1", "-vn", audio_path,
            ], capture_output=True, check=True)
            return audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"音频提取失败: {e}")
            return None

    def save_results(self, run_id: str) -> None:
        """保存语音结果"""
        if not self._events:
            logger.warning("没有事件可保存")
            return

        # 仅保留具有有效关键时刻的事件，并且每一项格式严格为 localSec 与 key_moment (无 text 字段)
        key_moment_events = []
        for event in self._events:
            key_moment = event.get("key_moment")
            if key_moment:
                key_moment_events.append({
                    "localSec": event.get("localSec"),
                    "key_moment": key_moment
                })

        # 调用存储器，解耦保存
        self._result_storage.save_key_moments(run_id, key_moment_events)

        # 保存完整文本
        full_text = " ".join(event.get("text", "") for event in self._events)
        self._result_storage.save_full_text(run_id, full_text)

        logger.info(f"语音结果已保存到 {run_id}")
