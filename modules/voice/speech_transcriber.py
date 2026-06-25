"""
语音转文字模块

使用 Whisper 进行语音转录，带后处理纠错和幻觉移除。
技术方案来自备份版本 A_DemoSrc1/VOICE/voice.py
"""
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger("module.voice.transcriber")

# Whisper 提示词（核电领域术语）
PROMPT_V4 = (
    "T1RPA034实验。1ES013V。浮片对齐。操作页数。"
    "请求监护。框架正确。KICK。SVideo。LCO306.6。"
    "1ES001VB。1ES013VSM3。D坑。报警。地坑隔离法。"
    "RPA34FU。1ES005。核对。确认。开启。基础状态。"
)

# ======================== 后处理纠错规则 ========================
CORRECTIONS = [
    # ---- 完整句子级纠错 ----
    ("操作页数。实验。确认。实验。没问题", "操作页数，好，四页，好，确认。四页没问题"),

    # ---- 基础状态 ----
    ("拍摄一下基础照片", "开始一下基础状态"),
    ("拍摄一下基础状态", "开始一下基础状态"),
    ("拍摄一下清楚状态", "开始一下基础状态"),
    ("拍摄下基础状态", "开始一下基础状态"),
    ("拍摄下基础照片", "开始一下基础状态"),
    ("排除下基础状态", "开始一下基础状态"),
    ("拍这下记录照片", "开始一下基础状态"),
    ("拍摄下清楚状态", "开始一下基础状态"),
    ("拍摄一下清录照片", "开始一下基础状态"),
    ("派出一下记录状态", "开始一下基础状态"),
    ("配置一下基础状态", "开始一下基础状态"),
    ("拍置下清录照片", "开始一下基础状态"),
    ("拍摄下进入状态", "开始一下基础状态"),

    # ---- RPA34FU ----
    ("RPR 34FU 34RP", "RPA34FU"), ("RPR34FU 34RP", "RPA34FU"),
    ("RPR34FU34RP", "RPA34FU"), ("RPA34FU34RP", "RPA34FU"),
    ("RP234F634RP", "RPA34FU"), ("RPR34FU，34RP", "RPA34FU"),
    ("RPA34FU，34RP", "RPA34FU"), ("RPA34FU。34RP", "RPA34FU"),
    ("RPA34FU。。", "RPA34FU，好，"), ("。34RP", ""),

    # ---- 设备编号 ----
    ("EES-013AB", "1ES013V"), ("EES-013AV", "1ES013V"),
    ("EES-011AB", "1ES011V"), ("EES013AV", "1ES013V"),
    ("0ES013AV", "1ES013V"), ("1ES01VB", "1ES011V"),
    ("1ES027I", "1ES027VB"), ("1ES021", "1ES02T"),
    ("1ES001PO", "1ES001PU"), ("1ES015", "1ES010"),
    ("1ES0PL", "1ES010"),
    ("EES-00", "1ES00"), ("EEX00", "1ES00"),
    ("ES00PO", "1ES005"), ("ES004", "1ES005"),
    ("ES002", "1ES005"), ("ES00P5", "1ES005"),
    ("1ES001VB。确认", "浮片对齐，好，确认"),

    # ---- KICK ----
    ("Keyboard上", "KICK中"), ("Keyboard中", "KICK中"), ("Keyboard", "KICK"),
    ("Key股中", "KICK中"), ("Key图中", "KICK中"), ("KEY图中", "KICK中"),
    ("KEY口中", "KICK中"), ("KEG中", "KICK中"), ("Key中", "KICK中"),
    ("KEY中", "KICK中"), ("K方", "KICK"),

    # ---- LCO306.6 ----
    ("lco3.6.6", "LCO306.6"), ("lc3.6.6", "LCO306.6"),
    ("LCO3.6.6", "LCO306.6"), ("FCO3.6.6", "LCO306.6"),
    ("FU3.6.6", "LCO306.6"), ("FCO306.6", "LCO306.6"),

    # ---- SVideo ----
    ("SV掉", "SVideo"), ("SV的欧", "SVideo"), ("S Video", "SVideo"), ("SV调", "SVideo"),
    ("SV掉了", "SVideo"), ("SV调了", "SVideo"), ("SV掉的", "SVideo"),
    ("SV叼", "SVideo"), ("SV 叼", "SVideo"),

    # ---- 浮片对齐 ----
    ("不断运行", "浮片对齐"), ("不对行", "浮片对齐"), ("不对应行", "浮片对齐"),
    ("不变形", "浮片对齐"), ("不对应性", "浮片对齐"), ("不对劲", "浮片对齐"),
    ("不动运行", "浮片对齐"), ("不对齐", "浮片对齐"), ("不定行", "浮片对齐"),
    ("不对。确认", "浮片对齐，好，确认"), ("富片对齐", "浮片对齐"),
    ("福片对齐", "浮片对齐"), ("扶片对齐", "浮片对齐"), ("对应行", "浮片对齐"),

    # ---- 操作页数 ----
    ("超弹变速", "操作页数"), ("超弹元素", "操作页数"), ("超弹叶数", "操作页数"),
    ("超强变速", "操作页数"), ("操作也数", "操作页数"), ("超点元素", "操作页数"),
    ("车标页数", "操作页数"),

    # ---- 控件 ----
    ("空间正确", "控件正确"), ("把空间推出了", "控件退出"), ("空间推出", "控件退出"), ("空间", "控件"),

    # ---- 请求监护 ----
    ("请求进入", "请求监护"), ("请求接入", "请求监护"),

    # ---- 其他术语 ----
    ("重按", "长按"), ("此按", "长按"), ("呈案", "长按"),
    ("好。合适。好。合适", "好，核实，好，核实"),
    ("合适正确", "核实正确"), ("合适", "核实"),
    ("业证", "验证"), ("试验", "实验"), ("执行T1RPA034", "进行T1RPA034"),
    ("四页画面", "实验画面"), ("四页方面", "实验画面"),
    ("4A方面", "实验画面"), ("4A画面", "实验画面"),
    ("4A", "四页"), ("4E", "四页"), ("4a", "四页"),
    ("注入条件", "出入条件"), ("处理条件", "出入条件"),
    ("免贴", "没问题"), ("免提", "没问题"),
    ("F6的", "2V的"), ("先下基础状态", "先是1V。基础状态"),
    ("先是1F2", "先是1V"), ("先下1V", "先是1V"),
    ("可以报警了", "报警"), ("入口。滑", "入口阀"), ("入口。法", "入口阀"),
    ("正常出发", "正常触发"), ("画面在确认", "画面再确认一遍"),
    ("可以进行", "可以执行"),
    ("手放", "收到"),

    # ---- 幻觉移除 ----
    ("请使用简体中文输出", ""), ("请不吝点赞", ""), ("订阅", ""),
    ("谢谢大家", ""), ("谢谢", ""),
]


def apply_corrections(text: str) -> str:
    """应用后处理纠错规则，按最长匹配优先"""
    corrected = text
    for wrong, right in sorted(CORRECTIONS, key=lambda x: len(x[0]), reverse=True):
        corrected = corrected.replace(wrong, right)
    return re.sub(r'[。，]{3,}', '。', corrected)


def remove_hallucinations(words):
    """移除连续重复的短词（Whisper 幻觉）"""
    if not words:
        return words
    cleaned, i = [], 0
    while i < len(words):
        w = words[i]
        j = i + 1
        while j < len(words) and words[j]["word"] == w["word"]:
            j += 1
        count = j - i
        if count >= 3 and len(w["word"]) <= 2:
            cleaned.append(words[i])
            cleaned.append(words[i + 1])
            i = j
        else:
            cleaned.append(w)
            i += 1
    return cleaned


class SpeechTranscriber:
    """
    语音转文字器

    使用 Whisper 模型进行语音转录，带后处理纠错。
    """

    def __init__(
        self,
        model_path: str = None,
        tokenizer_path: str = None,
        sample_rate: int = 16000,
    ):
        self.model_path = model_path or "large-v3"
        self.sample_rate = sample_rate
        self._model = None

    def _load_model(self):
        """加载 Whisper 模型"""
        if self._model is not None:
            return

        try:
            import whisper

            # 优先从本地路径加载
            local_path = self.model_path
            if os.path.isfile(local_path):
                logger.info(f"从本地加载 Whisper 模型: {local_path}")
                self._model = whisper.load_model(local_path)
            else:
                # 尝试从 models/voice/ 目录加载
                voice_dir = Path(__file__).parent.parent.parent / "models" / "voice"
                local_file = voice_dir / f"{self.model_path}.pt"
                if local_file.exists():
                    logger.info(f"从 models/voice 加载 Whisper 模型: {local_file}")
                    self._model = whisper.load_model(str(local_file))
                else:
                    logger.info(f"加载 Whisper 模型: {self.model_path}")
                    self._model = whisper.load_model(self.model_path)

            logger.info("Whisper 模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            raise

    def transcribe(self, audio_path: str, progress_callback=None, on_segment=None) -> List[Dict]:
        """
        转录音频文件

        Args:
            audio_path: 音频文件路径
            progress_callback: 进度回调函数 callback(current_sec, total_sec)

        Returns:
            词列表 [{"word": str, "start": float, "end": float}, ...]
        """
        self._load_model()

        try:
            import librosa
            import noisereduce as nr

            # 加载音频
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            audio_duration = len(audio) / sr

            # 去噪
            logger.info("音频去噪...")
            audio = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)

            # 检测语音结束位置
            speech_end = self._detect_speech_end(audio, sr)
            if speech_end < audio_duration - 1.0:
                logger.info(f"尾部静音: 语音结束于 {speech_end:.1f}s（总时长 {audio_duration:.1f}s）")
                effective_end = speech_end + 1.0
            else:
                effective_end = audio_duration

            logger.info(f"音频时长: {audio_duration:.1f}s, 有效时长: {effective_end:.1f}s")

            # 滑动窗口分段转录（与备份版本一致：window=25, overlap=5）
            words = self._transcribe_segmented(
                audio, sr,
                window_sec=25, overlap_sec=5,
                start_time=0, end_time=effective_end,
                progress_callback=progress_callback,
                on_segment=on_segment,
            )

            # 移除幻觉
            words = remove_hallucinations(words)

            # 应用纠错规则
            for w in words:
                w["word"] = apply_corrections(w["word"])

            logger.info(f"转录完成，共 {len(words)} 个词")
            return words

        except Exception as e:
            logger.error(f"转录失败: {e}", exc_info=True)
            return []

    def _transcribe_segmented(
        self,
        audio: np.ndarray,
        sr: int,
        window_sec: float = 25,
        overlap_sec: float = 5,
        start_time: float = 0,
        end_time: float = None,
        progress_callback=None,
        on_segment=None,
    ) -> List[Dict]:
        """滑动窗口分段转录"""
        if end_time is None:
            end_time = len(audio) / sr

        step = window_sec - overlap_sec
        if step <= 0:
            raise ValueError("window_sec 必须大于 overlap_sec")

        # 预生成窗口
        windows = []
        t = float(start_time)
        while t < end_time:
            end = min(t + window_sec, end_time)
            windows.append((t, end))
            t += step

        # 尾段太短则合并到前一段
        if len(windows) >= 2:
            last_s, last_e = windows[-1]
            if (last_e - last_s) < 6.0:
                prev_s, _ = windows[-2]
                windows[-2] = (prev_s, end_time)
                windows.pop()

        all_words = []

        for i, (t, end) in enumerate(windows, start=1):
            # 更新进度
            if progress_callback:
                progress_callback(t, end_time)

            # 切片音频
            start_sample = int(t * sr)
            end_sample = min(int(end * sr), len(audio))
            segment_audio = audio[start_sample:end_sample]

            # 跳过过短或静音段
            if len(segment_audio) < sr * 0.5:
                continue
            seg_rms = np.sqrt(np.mean(segment_audio ** 2))
            if seg_rms < 0.002:
                logger.debug(f"跳过静音段 {i}/{len(windows)} ({t:.1f}s-{end:.1f}s)")
                continue

            # 转录（与备份版本参数一致）
            try:
                result = self._model.transcribe(
                    segment_audio,
                    language="zh",
                    task="transcribe",
                    verbose=False,
                    word_timestamps=False,
                    initial_prompt=PROMPT_V4,
                    condition_on_previous_text=False,
                    temperature=0.0,
                    beam_size=5,
                    no_speech_threshold=0.6,
                )

                seg_words = []
                for seg in result.get("segments", []):
                    st = round(t + float(seg.get("start", 0)), 3)
                    et = round(t + float(seg.get("end", end - t)), 3)
                    txt = seg.get("text", "").strip()
                    # 繁体转简体
                    try:
                        from opencc import OpenCC
                        cc = OpenCC('t2s')
                        txt = cc.convert(txt)
                    except ImportError:
                        pass
                    if txt:
                        seg_words.append({"word": txt, "start": st, "end": et})

                # 重叠区去重
                if i > 1 and all_words:
                    mid = t + overlap_sec / 2
                    seg_words = [w for w in seg_words if w["start"] >= mid]
                    all_words = [w for w in all_words if w["start"] < mid]

                all_words.extend(seg_words)

                # 子窗口级进度更新（每段转录后立即报告）
                if progress_callback:
                    progress_callback(end, end_time)

                # 逐段回调：转录一段就推一段
                if on_segment and seg_words:
                    on_segment(seg_words, t, end, end_time)

            except Exception as e:
                logger.warning(f"片段转录失败 ({t:.1f}-{end:.1f}s): {e}")

        return all_words

    def _detect_speech_end(self, audio: np.ndarray, sr: int, frame_len=2048, hop=512, rms_threshold=0.005) -> float:
        """从音频尾部向前扫描，找到最后一个 RMS > threshold 的位置（秒）"""
        n_frames = 1 + (len(audio) - frame_len) // hop
        if n_frames <= 0:
            return len(audio) / sr
        rms = np.array([
            np.sqrt(np.mean(audio[i*hop : i*hop+frame_len] ** 2))
            for i in range(n_frames)
        ])
        last_active = n_frames - 1
        for i in range(n_frames - 1, -1, -1):
            if rms[i] > rms_threshold:
                last_active = i
                break
        end_sec = (last_active * hop + frame_len) / sr
        return min(end_sec, len(audio) / sr)
