"""
语音模块入口

继承 BaseModule，实现统一接口。
"""
import os
import logging

from core.base_module import BaseModule
from core.message_bus import MessageBus, MsgType
from core.frontend_sync import FrontendSync
from core.path_manager import PathConfig

from modules.voice.speech_transcriber import SpeechTranscriber
from modules.voice.intent_classifier import IntentClassifier
from modules.voice.result_storage import VoiceResultStorage

logger = logging.getLogger("module.voice")


class VoiceModule(BaseModule):
    """
    语音模块

    负责语音转文字和意图分类。
    """

    def __init__(
        self,
        bus: MessageBus,
        config: dict,
        paths: PathConfig,
        aggregator: FrontendSync,
    ):
        super().__init__(bus, config, paths, aggregator)
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

            # 初始化意图分类器
            self._intent_classifier = IntentClassifier(
                bus=self.bus,
                config=voice_config,
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
                """每段转录完成后的回调：分类 + 推送"""
                if not words:
                    return
                # 意图分类（逐段）
                events = self._intent_classifier.classify(words)
                # 推送到聚合器
                for event in events:
                    self.push_event("voice", {
                        "localSec": event.get("localSec", 0),
                        "text": event.get("text", ""),
                        "intent": event.get("intent", ""),
                        "key_moment": event.get("key_moment", ""),
                        "km_type": event.get("km_type", ""),
                        "device": event.get("device", ""),
                    })
                    self._events.append(event)
                # 更新进度
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

        # 保存关键事件
        self._result_storage.save_key_moments(run_id, self._events)

        # 保存完整文本
        full_text = " ".join(event.get("text", "") for event in self._events)
        self._result_storage.save_full_text(run_id, full_text)

        logger.info(f"语音结果已保存到 {run_id}")
