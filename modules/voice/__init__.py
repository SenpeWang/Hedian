"""
语音模块

负责：
- 语音转文字（Whisper）
- 意图分类
- 结果保存

使用方式：
    from modules.voice import VoiceModule
    module = VoiceModule(bus, config, paths, aggregator)
    module.start(video_path, run_id)
"""
from modules.voice.voice_module import VoiceModule
from modules.voice.speech_transcriber import SpeechTranscriber
from modules.voice.intent_classifier import IntentClassifier
from modules.voice.result_storage import VoiceResultStorage

__all__ = [
    "VoiceModule",
    "SpeechTranscriber",
    "IntentClassifier",
    "VoiceResultStorage",
]
