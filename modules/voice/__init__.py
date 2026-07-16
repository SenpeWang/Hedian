"""
语音模块

负责：
- 语音转文字（Whisper）
- 意图分类
- 结果保存

使用方式：
    from modules.voice import VoiceModule
    module = VoiceModule(event_bus, config, paths, display_buffer)
    module.start(video_path, run_id)
"""
from modules.voice.voice_module import VoiceModule
from modules.voice.speech_transcriber import SpeechTranscriber
from modules.voice.storage import VoiceResultStorage

__all__ = [
    "VoiceModule",
    "SpeechTranscriber",
    "VoiceResultStorage",
]
