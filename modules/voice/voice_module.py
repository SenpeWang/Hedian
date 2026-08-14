"""
语音模块入口.

继承 BaseModule，实现统一接口。
"""
import logging

from core.base_module import BaseModule
from core.event_bus import EventStream, EventTopic
from core.inference_stream import InferenceStream
from core.path_manager import PathManager

from modules.voice.speech_transcriber import (
    SpeechTranscriber,
    process_transcribed_words,
    normalize_devices_in_text,
)
from modules.voice.storage_voice import VoiceResultStorage

logger = logging.getLogger("module.voice")


class VoiceModule(BaseModule):
    """
    语音模块.

    负责语音转文字和意图分类。
    """

    def __init__(
        self,
        event_bus: EventStream,
        config: dict,
        paths: PathManager,
        inference_stream: InferenceStream,
    ):
        """初始化."""
        super().__init__(event_bus, config, paths, inference_stream)
        self._transcriber = None
        self._intent_classifier = None
        self._result_storage = None
        self._events = []

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
                device="cuda",
                torch_dtype=voice_config.get("torch_dtype"),
            )

            # 初始化结果存储
            self._result_storage = VoiceResultStorage(self.paths)

            logger.info("语音模块初始化完成")
            return True

        except Exception as e:
            logger.error(f"语音模块初始化失败: {e}", exc_info=True)
            return False

    def process_video(self, video_path: str) -> None:
        """处理视频，逐段转录、逐段分类、逐段推送."""
        try:
            # 提取音频
            audio_path = self._extract_audio(video_path)
            if not audio_path:
                logger.error("音频提取失败")
                return

            logger.info("开始语音转文字（逐段模式）...")

            def on_segment_done(words, seg_start, seg_end, total_dur):
                """每段转录完成后的回调."""
                if not words:
                    return
                events = process_transcribed_words(words, sentence_gap_sec=self.config.get("voice", {}).get("sentence_gap_sec", 0.6))
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
                        self.push_display("voice", {
                            "localSec": local_sec,
                            "tag": "text",
                            "data": {"text": display_text, "keys": keys}
                        })
                        self.inference_stream.update_module_snapshot("voice", {"latest_text": display_text})

                    # key_moment 既推推理流又推事件流
                    if key_moment:
                        self.push_display("voice", {
                            "localSec": local_sec,
                            "tag": "key_moment",
                            "data": {"key_moment": key_moment}
                        })
                        self.push_event(EventTopic.VOICE_KEY_MOMENT, {
                            "localSec": local_sec,
                            "key_moment": key_moment,
                        }, ts=local_sec)

                    # 保存归一化后的文本，确保 voice_full_text.json 中的设备码为纯英文数字
                    event["text"] = display_text if text else event.get("text", "")
                    self._events.append(event)
                
                # 实时增量落盘：每段转录产出事件后立即原子写盘，确保 evaluation 随时可提取
                if events and self._result_storage:
                    self._result_storage.save_results(self._run_id, self._events)

                self.update_progress(seg_end, total_dur)
                logger.debug("voice segment %.1f-%.1fs: %d events", seg_start, seg_end, len(events))

            def voice_progress(current, total_duration):
                """语音进度."""
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
        """从视频提取音频."""
        import subprocess

        # 使用当前推理的 run_id，确保音频文件与其他结果位于同一 result_dir
        audio_path = str(self.paths.get_result_path(
            run_id=self._run_id,
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
        """保存语音结果（委托给 VoiceResultStorage."""
        self._result_storage.save_results(run_id, self._events)
