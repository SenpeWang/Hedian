"""
语音模块

负责：
- 语音转文字（Qwen3-ASR）
- 意图分类
- 结果保存

使用方式：
    from modules.voice import VoiceModule
    module = VoiceModule(event_bus, config, paths, inference_stream)
    module.start(video_path, run_id)
"""
from modules.voice.voice_module import VoiceModule
from modules.voice.speech_transcriber import SpeechTranscriber
from modules.voice.storage_voice import VoiceResultStorage

__all__ = [
    "VoiceModule",
    "SpeechTranscriber",
    "VoiceResultStorage",
]
