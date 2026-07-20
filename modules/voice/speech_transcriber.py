"""
语音转文字模块

使用 Qwen3-ASR 进行语音转录，带后处理纠错和幻觉过滤。
"""
import os
import re
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger("module.voice.transcriber")


# 后处理纠错规则
CORRECTIONS = [
    # 请求监护
    ("请调监控", "请求监护"), ("请调监护", "请求监护"),
    ("请台监护", "请求监护"), ("请台军务", "请求监护"),
    ("请结束", "请求监护"), ("请求进入", "请求监护"), ("请求接入", "请求监护"),
    ("台军务", "请求监护"),

    # 核实/核对
    ("合适正确", "核实正确"), ("合适", "核实"),

    # 执行
    ("可以进行", "可以执行"),

    # 常见术语
    ("试验", "实验"), ("业证", "验证"),
    ("注入条件", "出入条件"), ("处理条件", "出入条件"),
    ("可以报警了", "报警"), ("正常出发", "正常触发"),

    # 幻觉过滤
    ("请使用简体中文输出", ""), ("请不吝点赞", ""), ("订阅", ""),
    ("谢谢大家", ""), ("谢谢", ""),
]


def apply_corrections(text: str) -> str:
    """应用后处理纠错规则，按最长匹配优先"""
    corrected = text
    for wrong, right in sorted(CORRECTIONS, key=lambda x: len(x[0]), reverse=True):
        corrected = corrected.replace(wrong, right)
    return re.sub(r'[。，]{3,}', '。', corrected)


# 已移除无用的 ASR 幻觉去重函数



# ======================== Qwen3-ASR 辅助函数 ========================
def _read_attr_or_key(item, *names, default=None):
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default

def _safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def normalize_qwen_results(results, duration_seconds=0.0):
    if results is None:
        return []
    if not isinstance(results, (list, tuple)):
        results = [results]

    segments = []
    for result in results:
        text = str(_read_attr_or_key(result, "text", default="") or "").strip()
        raw_stamps = (
            _read_attr_or_key(result, "time_stamps")
            or _read_attr_or_key(result, "timestamps")
            or _read_attr_or_key(result, "words")
            or []
        )
        words = []
        for stamp in raw_stamps:
            word_text = str(
                _read_attr_or_key(stamp, "text", "word", "token", default="") or ""
            ).strip()
            if not word_text:
                continue
            start = _safe_float(_read_attr_or_key(stamp, "start_time", "start", "begin_time", default=0.0))
            end = _safe_float(_read_attr_or_key(stamp, "end_time", "end", "finish_time", default=start))
            words.append({
                "start": float(start),
                "end": float(end if end >= start else start),
                "word": word_text,
            })

        if words:
            start = min(word["start"] for word in words)
            end = max(word["end"] for word in words)
            if not text:
                text = "".join(word["word"] for word in words)
        else:
            start = 0.0
            end = float(duration_seconds or 0.0)

        if not text:
            continue

        item = {
            "start": float(start),
            "end": float(end if end >= start else start),
            "text": text,
        }
        if words:
            item["words"] = words
        segments.append(item)
    return segments


class SpeechTranscriber:
    """
    语音转文字器

    使用 Qwen3-ASR 模型进行语音转录，带后处理纠错。
    """

    def __init__(
        self,
        model_path: str = None,
        aligner_path: str = None,
        asr_engine: str = "qwen3",
        sample_rate: int = 16000,
        device: str = "cpu",
        torch_dtype: str = None,
    ):
        self.model_path = model_path
        self.aligner_path = aligner_path
        self.asr_engine = asr_engine
        self.sample_rate = sample_rate
        self.device = device
        self.torch_dtype = torch_dtype
        self._model = None

    def _load_model(self):
        """加载 ASR 模型"""
        if self._model is not None:
            return

        try:
            import torch
            from qwen_asr import Qwen3ASRModel

            # 配置推理参数
            device_map = "cuda:0" if self.device == "cuda" else "cpu"
            torch_type = getattr(torch, self.torch_dtype) if self.torch_dtype else (torch.bfloat16 if self.device == "cuda" else torch.float32)

            kwargs = {
                "dtype": torch_type,
                "device_map": device_map,
                "num_beams": 5,
                "do_sample": False,
                "repetition_penalty": 1.1,
                "max_new_tokens": 4096,
                "max_inference_batch_size": 8,
            }

            # 如果有 aligner 并且路径存在，则使用它
            if self.aligner_path and os.path.exists(self.aligner_path):
                kwargs["forced_aligner"] = self.aligner_path
                kwargs["forced_aligner_kwargs"] = {
                    "dtype": torch_type,
                    "device_map": device_map,
                }
                logger.info(f"开启 Word-level Aligner 对齐器: {self.aligner_path}")
            else:
                logger.warning("未检测到或未配置对齐器 Aligner 路径，将仅使用 ASR 段级时间戳")

            logger.info(f"正在从本地加载 Qwen3-ASR 模型: {self.model_path}")
            self._model = Qwen3ASRModel.from_pretrained(self.model_path, **kwargs)
            logger.info("Qwen3-ASR 模型加载完成")
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

            # 分段转录(5s重叠)
            words = self._transcribe_segments(
                audio, sr,
                window_sec=20, overlap_sec=5,
                start_time=0, end_time=effective_end,
                progress_callback=progress_callback,
                on_segment=on_segment,
            )

            logger.info(f"转录完成，共 {len(words)} 个词")
            return words

        except Exception as e:
            logger.error(f"转录失败: {e}", exc_info=True)
            return []

    def _transcribe_segments(
        self,
        audio: np.ndarray,
        sr: int,
        window_sec: float = 20,
        overlap_sec: float = 0,
        start_time: float = 0,
        end_time: float = None,
        progress_callback=None,
        on_segment=None,
    ) -> List[Dict]:
        """分段转录(带重叠)，每段独立转录，midpoint去重"""
        if end_time is None:
            end_time = len(audio) / sr

        step = window_sec - overlap_sec
        if step <= 0:
            step = window_sec

        windows = []
        t = float(start_time)
        while t < end_time:
            cur_end = min(t + window_sec, end_time)
            windows.append((t, cur_end))
            t += step

        all_words = []

        for i, (t, cur_end) in enumerate(windows, start=1):
            if progress_callback:
                progress_callback(t, end_time)

            start_sample = int(t * sr)
            end_sample = min(int(cur_end * sr), len(audio))
            seg_audio = audio[start_sample:end_sample]

            if len(seg_audio) < sr * 0.5:
                continue
            seg_rms = np.sqrt(np.mean(seg_audio ** 2))
            if seg_rms < 0.002:
                continue

            try:
                results = self._model.transcribe(
                    audio=(seg_audio, sr),
                    language="Chinese",
                    return_time_stamps=bool(self.aligner_path and os.path.exists(self.aligner_path)),
                )
                normalized = normalize_qwen_results(results, duration_seconds=len(seg_audio)/sr)

                seg_words = []
                for seg in normalized:
                    txt = seg.get("text", "").strip()
                    try:
                        from opencc import OpenCC
                        cc = OpenCC("t2s")
                        txt = cc.convert(txt)
                    except ImportError:
                        pass
                    if txt:
                        if "words" in seg and seg["words"]:
                            for w in seg["words"]:
                                seg_words.append({
                                    "start": round(t + float(w.get("start", 0.0)), 3),
                                    "end": round(t + float(w.get("end", 0.0)), 3),
                                    "word": w.get("word") or w.get("text") or "",
                                })
                        else:
                            st = round(t + float(seg.get("start", 0.0)), 3)
                            et = round(t + float(seg.get("end", 0.0)), 3)
                            seg_words.append({
                                "word": txt,
                                "start": st,
                                "end": et
                            })

                # midpoint去重: 平铺模式下，只需直接保留字词起始时间大于 midpoint 的项
                if overlap_sec > 0 and i > 1 and all_words:
                    mid = t + overlap_sec / 2
                    seg_words = [w for w in seg_words if w["start"] >= mid]

                all_words.extend(seg_words)

                if progress_callback:
                    progress_callback(cur_end, end_time)

                if seg_words and on_segment:
                    # 单字无须应用多字纠错，直接将平铺词表传递给 downstream 回调即可
                    on_segment(seg_words, t, cur_end, end_time)

            except Exception as e:
                logger.warning("segment failed (%.1f-%.1fs): %s", t, cur_end, e)

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


# ======================== 文本归一化与关键词提取 ========================

# 严格 9 字码模式：1位数字 + 3位字母 + 3位数字 + 2位字母 = 9 字符
# 按用户要求：九字码一定是九位数,否则就是识别错了,不接受其他长度
# 合法归一化后的设备码类型正则匹配（支持 1EAS013VB, T1RPA034, RPA34FU, LCO3.6.6, SM3）
NORM_DEVICE_PATTERN = re.compile(
    r"("
    r"1EAS\d{3}[A-Z]{2}"
    r"|T1RPA\d{3}"
    r"|RPA\d{2}[A-Z]{2}"
    r"|LCO[0-9\.]+"
    r"|SM\d+"
    r")",
    re.IGNORECASE
)

CN_DIGITS = {
    "零": 0, "〇": 0, "洞": 0,
    "一": 1, "幺": 1, "腰": 1,
    "二": 2, "两": 2,
    "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}

CN_UNITS = {
    "十": 10, "百": 100, "千": 1000,
}

LETTER_WORDS = {
    "阿尔": "R", "艾儿": "R",
    "艾斯": "S", "爱斯": "S",
    "维": "V", "威": "V", "微": "V",
    "批": "P", "屁": "P", "皮": "P",
    "诶": "A",
}

def _convert_cn_number(token):
    if not token:
        return ""
    if any(ch in CN_UNITS for ch in token):
        total = 0
        current = 0
        for ch in token:
            if ch in CN_DIGITS:
                current = CN_DIGITS[ch]
            elif ch in CN_UNITS:
                unit = CN_UNITS[ch]
                if current == 0:
                    current = 1
                total += current * unit
                current = 0
        total += current
        return str(total)
    chars = []
    for ch in token:
        if ch in CN_DIGITS:
            chars.append(str(CN_DIGITS[ch]))
    return "".join(chars)

def normalize_spoken_text(text):
    """把口语化的设备码文本归一化为纯英文+数字格式"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).upper()
    
    # 替换中文“点”为小数点“.”
    text = text.replace("点", ".")
    
    for word, letter in LETTER_WORDS.items():
        text = text.replace(word, letter)
    text = re.sub(
        r"[零〇洞一幺腰二两三四五六七八九十百千]+",
        lambda m: _convert_cn_number(m.group(0)),
        text,
    )
    # 常用音译前缀和数字归一化
    text = text.replace("11ES", "1EAS")
    text = text.replace("1ES", "1EAS")
    text = text.replace("EES", "1EAS").replace("EEX", "1EAS")
    # 用负向后行断言，避免对已归一化的 "1EAS" 再次添加前缀变成 "11EAS"
    text = re.sub(r"(?<![1-9])EAS", "1EAS", text)
    text = text.replace("ES", "1EAS")
    
    # RPR 纠错归一为 RPA
    text = text.replace("RPR", "RPA")
    
    # 只保留字母数字和小数点
    return re.sub(r"[^A-Z0-9\.]", "", text)


# 中文数字字符（用于宽松匹配原文本中的设备码片段）
_CN_DIGITS = "零〇洞一幺腰二两三四五六七八九"

# 宽松设备码匹配模式：在原文本中识别各种可能的设备码片段（包括 EAS/ES/EES/EEX, TRPA/T1RPA/RPA/RPR, LCO, SM）
LOOSE_DEVICE_PATTERN = re.compile(
    rf"(?:"
    rf"[1-9{_CN_DIGITS}]?EAS[a-zA-Z0-9{_CN_DIGITS}]+"
    rf"|[1-9{_CN_DIGITS}]?ES[a-zA-Z0-9{_CN_DIGITS}]+"
    rf"|EES[a-zA-Z0-9{_CN_DIGITS}]+"
    rf"|EEX[a-zA-Z0-9{_CN_DIGITS}]+"
    rf"|T?[1-9{_CN_DIGITS}]?RPA[a-zA-Z0-9{_CN_DIGITS}]+"
    rf"|T?[1-9{_CN_DIGITS}]?RPR[a-zA-Z0-9{_CN_DIGITS}]+"
    rf"|LCO[a-zA-Z0-9{_CN_DIGITS}\.点]+"
    rf"|S\s*M\s*[a-zA-Z0-9{_CN_DIGITS}]+"
    rf")",
    re.IGNORECASE
)


def normalize_devices_in_text(text: str) -> str:
    """
    把文本中的设备码片段归一化为纯英文+数字格式（保留其他中文）。
    严格校验不同类型设备码的正确长度：
      - 九字码 (1EAS) 必须为 9 字符
      - T1RPA 必须为 8 字符
      - RPA 必须为 7 字符
      - LCO 必须为 8 字符
      - SM 必须为 3 字符
    """
    if not text:
        return ""
    
    # 预处理：去除所有的空格和制表符，以匹配 ASR 偶发的空格分割（如 "E E S" 或 "T 1 R P A"）
    cleaned_text = re.sub(r"\s+", "", text)
    
    def _norm(m):
        normalized = normalize_spoken_text(m.group(0))
        # 根据前缀严格校验其各自正确的字符长度
        is_valid = False
        if normalized.startswith("1EAS"):
            is_valid = (len(normalized) == 9)
        elif normalized.startswith("T1RPA"):
            is_valid = (len(normalized) == 8)
        elif normalized.startswith("RPA"):
            is_valid = (len(normalized) == 7)
        elif normalized.startswith("LCO"):
            is_valid = (len(normalized) == 8)
        elif normalized.startswith("SM"):
            is_valid = (len(normalized) == 3)
            
        if is_valid:
            return normalized
        return m.group(0)  # 长度或格式不符，不予归一化，保留原样
        
    return LOOSE_DEVICE_PATTERN.sub(_norm, cleaned_text)


# 拼音模糊编辑距离（Levenshtein 距离）匹配算法
def match_keyword_by_pinyin_levenshtein(text: str, keyword: str, max_distance: int = 1) -> bool:
    """
    利用拼音序列滑动窗口编辑距离，在 ASR 文本中寻找发音高度相似的关键词。
    - max_distance: 允许的最大单字拼音不同数量，1 表示最多允许错一个字音
    - 相比 Syllable Intersection 比例比对，能极大地避免字词交集造成的误判，并提供优秀的 ASR 音似容错
    """
    try:
        import pypinyin
    except ImportError:
        return False

    keyword_pinyins = pypinyin.lazy_pinyin(keyword)
    text_pinyins = pypinyin.lazy_pinyin(text)

    n, m = len(text_pinyins), len(keyword_pinyins)
    if n < m:
        return False

    # 滑动窗口比对拼音序列
    for i in range(n - m + 1):
        sub_pinyins = text_pinyins[i:i+m]
        dist = sum(1 for p1, p2 in zip(sub_pinyins, keyword_pinyins) if p1 != p2)
        if dist <= max_distance:
            return True
    return False


def match_keyword_by_pinyin(text: str, keyword: str) -> bool:
    """
    综合文本精准匹配与拼音编辑距离匹配。
    - 对于长度 <= 3 的短关键词（如“监护”、“核对”、“收到”），拼音必须严格 100% 匹配（max_distance = 0）
    - 对于长度 >= 4 的长关键词（如“请求监护”、“信息通报”），允许 1 位字音偏差（max_distance = 1）以容忍 ASR 偶发的字词偏差
    """
    if keyword in text:
        return True
    
    # 动态设定最大编辑距离
    max_dist = 1 if len(keyword) >= 4 else 0
    return match_keyword_by_pinyin_levenshtein(text, keyword, max_distance=max_dist)


# 关键词识别：(关键词, 标签) 列表 - 标签是发送给规则模块的 key_moment 值
# 变体词归一化到标准标签，避免重复保存（如"监护"→"请求监护"，"核实"→"核对"）
KEYWORD_LABELS = [
    ("请求监护", "请求监护"),
    ("监护",     "请求监护"),
    ("执行",     "执行"),
    ("核对",     "核对"),
    ("核实",     "核对"),
    ("信息通报", "信息通报"),
    ("信息通告", "信息通报"),
    ("通报完毕", "通报完毕"),
    ("通告完毕", "通报完毕"),
    ("收到",     "收到"),
]


def process_transcribed_words(words: List[Dict], sentence_gap_sec: float = 1.0) -> List[Dict]:
    """
    对字/词按停顿切分出段落，并提取关键事件。

    - 每个句子推送一条事件（带完整 text）
    - 检测到的关键词和 9 字符设备码作为 key_moment
    - 没有关键词的句子也推送（key_moment 为空字符串）
    """
    if not words:
        return []

    words = [w for w in words if w is not None and w.get("word")]
    if not words:
        return []

    # 分句：按停顿切分
    sentences = []
    cur = []
    for w in words:
        if cur and w["start"] - cur[-1]["end"] > sentence_gap_sec:
            sentences.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        sentences.append(cur)

    events = []
    for sent_words in sentences:
        raw_text = "".join(w["word"] for w in sent_words)
        if not raw_text.strip():
            continue

        text = apply_corrections(raw_text) # 关键纠错：让比对和最终通知文本均使用纠错后的规范汉字，以便能够正确匹配并上报“请求监护”等关键事件
        audio_ts = round(sent_words[0]["start"], 2)

        # 设备码提取：用 finditer 拆分串联码，提取所有合法 9 字符设备码
        norm_text = normalize_spoken_text(text)
        devices = [m.group(1) for m in NORM_DEVICE_PATTERN.finditer(norm_text)]

        # 收集所有匹配的关键字（严格拼音匹配，要求所有音节都出现）
        found_keywords = []
        for device in devices:
            found_keywords.append(device)
        for keyword, label in KEYWORD_LABELS:
            if keyword == "收到":
                # "收到" 容易在长段落中误报，仅在完全一致或满足拼音相似度时才上报
                if text.strip() == "收到" or match_keyword_by_pinyin(text, keyword):
                    found_keywords.append(label)
            elif match_keyword_by_pinyin(text, keyword):
                found_keywords.append(label)

        # 去重（保持顺序）
        seen = set()
        unique_keywords = []
        for km in found_keywords:
            if km not in seen:
                seen.add(km)
                unique_keywords.append(km)

        # 每句推一条事件（带完整 text + key_moment 列表）
        # 如果有关键词/设备码，第一个事件带 text，后续事件只带 key_moment
        if unique_keywords:
            first = True
            for km in unique_keywords:
                ev = {"localSec": audio_ts, "key_moment": km}
                if first:
                    ev["text"] = text
                    first = False
                events.append(ev)
        else:
            # 没有关键词也推送完整文本（用于推理流展示）
            events.append({"localSec": audio_ts, "text": text, "key_moment": ""})

    return events
