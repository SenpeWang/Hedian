"""语音模块：Qwen3-ASR 语音转文字 + 意图分类 + 结果保存."""
from modules.voice.voice_module import VoiceModule
from modules.voice.speech_transcriber import SpeechTranscriber
from modules.voice.storage_voice import VoiceResultStorage

__all__ = [
    "VoiceModule",
    "SpeechTranscriber",
    "VoiceResultStorage",
]
