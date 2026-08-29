"""语音转文字模块：基于 Qwen3-ASR 转录音频，带后处理纠错与幻觉过滤.

分段转录音频并做重叠去重，随后应用拼音纠错、设备码归一化与关键词匹配，
产出带时间戳的词表与关键事件（key_moment）。
"""
import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

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
    """应用后处理纠错规则，按最长匹配优先替换并压缩连续标点.

    Args:
        text: ASR 原始识别文本.

    Returns:
        应用全部纠错规则与标点压缩后的文本.
    """
    corrected = text
    for wrong, right in sorted(
        CORRECTIONS,
        key=lambda correction: len(correction[0]),
        reverse=True,
    ):
        corrected = corrected.replace(wrong, right)
    return re.sub(r'[。，]{3,}', '。', corrected)


# ======================== Qwen3-ASR 辅助函数 ========================
def _read_attr_or_key(item: Any, *names: str, default: Any = None) -> Any:
    """按候选名依次读取字典键或对象属性.

    Args:
        item: 待读取的字典或对象.
        *names: 候选字段名，按顺序尝试.
        default: 全部候选都不存在时返回的默认值.

    Returns:
        第一个命中的字段值，未命中时返回 default.
    """
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """把任意输入安全转换为浮点数，空值与非法输入回退到默认值.

    Args:
        value: 待转换的值，允许为 None 或空字符串.
        default: 转换失败或输入为空时返回的默认值.

    Returns:
        转换得到的浮点数.
    """
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_qwen_results(
    results: Any,
    duration_seconds: float = 0.0,
) -> List[Dict[str, Any]]:
    """把 Qwen3-ASR 返回结果统一归一化为段级字典列表.

    兼容列表/单对象两种返回形态，并按 time_stamps/timestamps/words 三个
    候选字段读取字级时间戳；无字级信息时退回段级起止时间。

    Args:
        results: Qwen3-ASR 的原始返回结果.
        duration_seconds: 音频总时长（秒），用于无字级信息时兜底段结束时间.

    Returns:
        段级结果列表，每项含 start/end/text，有字级信息时附加 words.
    """
    if results is None:
        return []
    if not isinstance(results, (list, tuple)):
        results = [results]

    segments = []
    for result in results:
        text = str(_read_attr_or_key(result, "text", default="") or "").strip()
        raw_timestamps = (
            _read_attr_or_key(result, "time_stamps")
            or _read_attr_or_key(result, "timestamps")
            or _read_attr_or_key(result, "words")
            or []
        )
        words = []
        for stamp in raw_timestamps:
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
            start = min(word_item["start"] for word_item in words)
            end = max(word_item["end"] for word_item in words)
            if not text:
                text = "".join(word_item["word"] for word_item in words)
        else:
            start = 0.0
            end = float(duration_seconds or 0.0)

        if not text:
            continue

        segment_result = {
            "start": float(start),
            "end": float(end if end >= start else start),
            "text": text,
        }
        if words:
            segment_result["words"] = words
        segments.append(segment_result)
    return segments


class SpeechTranscriber:
    """基于 Qwen3-ASR 的语音转文字器.

    负责模型加载、分段转录与去重，结果的后处理纠错在文本归一化函数中完成。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        aligner_path: Optional[str] = None,
        sample_rate: int = 16000,
        device: str = "cuda",
        torch_dtype: Optional[str] = None,
    ) -> None:
        """初始化转录器配置，模型在此阶段不会加载.

        Args:
            model_path: Qwen3-ASR 模型本地路径.
            aligner_path: Word-level Aligner 路径，存在时启用字级时间戳.
            sample_rate: 目标采样率（Hz）.
            device: 推理设备，"cuda" 或 "cpu".
            torch_dtype: PyTorch 数据类型名，为 None 时按设备自动选择.
        """
        self.model_path = model_path
        self.aligner_path = aligner_path
        self.sample_rate = sample_rate
        self.device = device
        self.torch_dtype = torch_dtype
        self._model: Any = None

    def _load_model(self) -> None:
        """加载 ASR 模型，已加载时直接返回.

        配置 beam search 与惩罚参数；对齐器路径存在且可访问时额外启用
        强制对齐以产出字级时间戳。
        """
        if self._model is not None:
            return

        try:
            import torch
            from qwen_asr import Qwen3ASRModel

            # 配置推理参数
            device_map = "cuda:0" if self.device == "cuda" else "cpu"
            torch_dtype = getattr(torch, self.torch_dtype) if self.torch_dtype else (torch.bfloat16 if self.device == "cuda" else torch.float32)

            kwargs = {
                "dtype": torch_dtype,
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
                    "dtype": torch_dtype,
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

    def transcribe(
        self,
        audio_path: str,
        progress_callback: Optional[Any] = None,
        on_segment: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """转录音频文件并返回带时间戳的词列表.

        先加载并降噪音频、定位语音结束位置，再按滑动窗口分段转录，
        可选地通过回调报告进度或逐段回调产出的词表。

        Args:
            audio_path: 音频文件路径.
            progress_callback: 进度回调 callback(current_sec, total_sec)，可为 None.
            on_segment: 逐段回调 callback(words, seg_start, seg_end, total_sec)，可为 None.

        Returns:
            词列表，每项含 word/start/end；转录失败时返回空列表.
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
        end_time: Optional[float] = None,
        progress_callback: Optional[Any] = None,
        on_segment: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """按滑动窗口分段转录，逐段去重后拼接为完整词表.

        重叠窗口按 midpoint 去重，避免重复词；过短或过静的窗口直接跳过，
        单段转录异常只告警不中断整体流程。

        Args:
            audio: 一维音频波形数组.
            sr: 采样率（Hz）.
            window_sec: 窗口时长（秒）.
            overlap_sec: 相邻窗口重叠时长（秒），0 表示无重叠.
            start_time: 首个窗口的起始时间（秒）.
            end_time: 转录结束时间（秒），为 None 时取音频实际时长.
            progress_callback: 进度回调 callback(current_sec, total_sec)，可为 None.
            on_segment: 逐段回调 callback(words, seg_start, seg_end, total_sec)，可为 None.

        Returns:
            按时间顺序拼接的词列表，每项含 word/start/end.
        """
        if end_time is None:
            end_time = len(audio) / sr

        step = window_sec - overlap_sec
        if step <= 0:
            step = window_sec

        windows = []
        window_start = float(start_time)
        while window_start < end_time:
            window_end = min(window_start + window_sec, end_time)
            windows.append((window_start, window_end))
            window_start += step

        all_words = []

        for segment_index, (window_start, window_end) in enumerate(windows, start=1):
            if progress_callback:
                progress_callback(window_start, end_time)

            start_sample = int(window_start * sr)
            end_sample = min(int(window_end * sr), len(audio))
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) < sr * 0.5:
                continue
            segment_rms = np.sqrt(np.mean(segment_audio ** 2))
            if segment_rms < 0.002:
                continue

            try:
                results = self._model.transcribe(
                    audio=(segment_audio, sr),
                    language="Chinese",
                    return_time_stamps=bool(
                        self.aligner_path and os.path.exists(self.aligner_path)
                    ),
                )
                normalized = normalize_qwen_results(
                    results, duration_seconds=len(segment_audio) / sr
                )

                try:
                    from opencc import OpenCC
                    converter = OpenCC("t2s")
                except ImportError:
                    converter = None

                segment_words = []
                for segment in normalized:
                    text = segment.get("text", "").strip()
                    if converter is not None:
                        text = converter.convert(text)
                    if text:
                        if "words" in segment and segment["words"]:
                            for word_item in segment["words"]:
                                segment_words.append({
                                    "start": round(
                                        window_start + float(word_item.get("start", 0.0)), 3
                                    ),
                                    "end": round(
                                        window_start + float(word_item.get("end", 0.0)), 3
                                    ),
                                    "word": word_item.get("word")
                                    or word_item.get("text")
                                    or "",
                                })
                        else:
                            start_sec = round(
                                window_start + float(segment.get("start", 0.0)), 3
                            )
                            end_sec = round(
                                window_start + float(segment.get("end", 0.0)), 3
                            )
                            segment_words.append({
                                "word": text,
                                "start": start_sec,
                                "end": end_sec
                            })

                # midpoint去重: 平铺模式下，只需直接保留字词起始时间大于 midpoint 的项
                if overlap_sec > 0 and segment_index > 1 and all_words:
                    mid = window_start + overlap_sec / 2
                    segment_words = [
                        word_item for word_item in segment_words
                        if word_item["start"] >= mid
                    ]

                all_words.extend(segment_words)

                if progress_callback:
                    progress_callback(window_end, end_time)

                if segment_words and on_segment:
                    # 单字无须应用多字纠错，直接将平铺词表传递给 downstream 回调即可
                    on_segment(segment_words, window_start, window_end, end_time)

            except Exception as e:
                logger.warning("segment failed (%.1f-%.1fs): %s", window_start, window_end, e)

        return all_words

    def _detect_speech_end(
        self,
        audio: np.ndarray,
        sr: int,
        frame_len: int = 2048,
        hop: int = 512,
        rms_threshold: float = 0.005,
    ) -> float:
        """从音频尾部向前扫描，定位最后一个有效语音帧的结束时间（秒）.

        Args:
            audio: 一维音频波形数组.
            sr: 采样率（Hz）.
            frame_len: 帧长（样本数）.
            hop: 帧移（样本数）.
            rms_threshold: 判定为语音的 RMS 能量阈值.

        Returns:
            语音结束时间（秒），上限为音频总时长.
        """
        n_frames = 1 + (len(audio) - frame_len) // hop
        if n_frames <= 0:
            return len(audio) / sr
        rms = np.array([
            np.sqrt(np.mean(audio[frame_index*hop : frame_index*hop+frame_len] ** 2))
            for frame_index in range(n_frames)
        ])
        last_active = n_frames - 1
        for frame_index in range(n_frames - 1, -1, -1):
            if rms[frame_index] > rms_threshold:
                last_active = frame_index
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


def _convert_cn_number(token: str) -> str:
    """把中文数字串转换为阿拉伯数字字符串.

    含十/百/千等量词时按位权累加（如"一百二十三"→"123"），纯个位数时
    逐字映射（如"幺三五"→"135"）；无法识别的字符直接丢弃。

    Args:
        token: 中文数字文本片段.

    Returns:
        转换后的阿拉伯数字字符串，无有效字符时为空字符串.
    """
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


def normalize_spoken_text(text: str) -> str:
    """把口语化的设备码文本归一化为纯英文+数字格式.

    依次做全角转半角、大小写统一、小数点归一、字母音译还原、中文数字
    转阿拉伯数字、前缀补齐与 RPR→RPA 纠错，最后只保留字母数字和小数点。

    Args:
        text: 可能含中文数字与音译字母的设备码文本.

    Returns:
        归一化后的设备码文本；输入为空时返回空字符串.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).upper()

    # 替换中文“点”为小数点“.”
    text = text.replace("点", ".")

    for word, letter in LETTER_WORDS.items():
        text = text.replace(word, letter)
    text = re.sub(
        r"[零〇洞一幺腰二两三四五六七八九十百千]+",
        lambda match: _convert_cn_number(match.group(0)),
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
    """把文本中的设备码片段归一化为纯英文+数字格式（保留其他中文）.

    严格校验不同类型设备码的正确长度：
      - 九字码 (1EAS) 必须为 9 字符
      - T1RPA 必须为 8 字符
      - RPA 必须为 7 字符
      - LCO 必须为 8 字符
      - SM 必须为 3 字符

    Args:
        text: 可能混有设备码片段的整句文本.

    Returns:
        设备码已被替换为规范形式的文本；输入为空时返回空字符串.
    """
    if not text:
        return ""

    # 预处理：去除所有的空格和制表符，以匹配 ASR 偶发的空格分割（如 "E E S" 或 "T 1 R P A"）
    cleaned_text = re.sub(r"\s+", "", text)

    def _normalize_device_code(match):
        """归一化单个设备码匹配片段并校验其长度合法性.

        Args:
            match: 宽松设备码模式的单个匹配对象.

        Returns:
            长度合法时返回归一化结果，否则原样返回.
        """
        normalized = normalize_spoken_text(match.group(0))
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
        return match.group(0)  # 长度或格式不符，不予归一化，保留原样

    return LOOSE_DEVICE_PATTERN.sub(_normalize_device_code, cleaned_text)


# 拼音模糊编辑距离（Levenshtein 距离）匹配算法
def match_keyword_by_pinyin_levenshtein(
    text: str,
    keyword: str,
    max_distance: int = 1,
) -> bool:
    """利用拼音序列滑动窗口编辑距离，在 ASR 文本中寻找发音高度相似的关键词.

    相比音节交集比例比对，能极大避免字词交集造成的误判，并提供优秀的 ASR 音似容错.

    Args:
        text: 待匹配的 ASR 识别文本.
        keyword: 目标关键词.
        max_distance: 允许的最大单字拼音不同数量，1 表示最多允许错一个字音.

    Returns:
        滑动窗口内存在拼音编辑距离不超过 max_distance 的子序列时返回 True.
    """
    try:
        import pypinyin
    except ImportError:
        return False

    keyword_pinyins = pypinyin.lazy_pinyin(keyword)
    text_pinyins = pypinyin.lazy_pinyin(text)

    text_count, keyword_count = len(text_pinyins), len(keyword_pinyins)
    if text_count < keyword_count:
        return False

    # 滑动窗口比对拼音序列
    for window_offset in range(text_count - keyword_count + 1):
        sub_pinyins = text_pinyins[window_offset:window_offset+keyword_count]
        distance = sum(1 for p1, p2 in zip(sub_pinyins, keyword_pinyins) if p1 != p2)
        if distance <= max_distance:
            return True
    return False


def match_keyword_by_pinyin(text: str, keyword: str) -> bool:
    """综合文本精准匹配与拼音编辑距离匹配.

    短关键词（长度 <= 3，如"监护"、"核对"、"收到"）要求拼音严格 100% 匹配；
    长关键词（长度 >= 4，如"请求监护"、"信息通报"）允许 1 位字音偏差以容忍
    ASR 偶发的字词偏差.

    Args:
        text: 待匹配的 ASR 识别文本.
        keyword: 目标关键词.

    Returns:
        文本包含关键词原文，或存在符合长度规则编辑距离的拼音相似子序列时返回 True.
    """
    if keyword in text:
        return True

    # 动态设定最大编辑距离
    max_distance = 1 if len(keyword) >= 4 else 0
    return match_keyword_by_pinyin_levenshtein(text, keyword, max_distance=max_distance)


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


def process_transcribed_words(
    words: List[Dict[str, Any]],
    sentence_gap_sec: float = 1.0,
) -> List[Dict[str, Any]]:
    """对字/词按停顿切分出句子，并提取关键事件.

    每个句子推送一条事件（带完整 text）；检测到的关键词与 9 字符设备码作为
    key_moment；没有关键词的句子也推送（key_moment 为空字符串）.

    Args:
        words: 带 start/end/word 字段的时间戳词表.
        sentence_gap_sec: 判定句子边界的词间停顿阈值（秒）.

    Returns:
        关键事件列表，每项包含 localSec、key_moment、keys，首条含 text.
    """
    if not words:
        return []

    words = [
        word_item for word_item in words
        if word_item is not None and word_item.get("word")
    ]
    if not words:
        return []

    # 分句：按停顿切分
    sentences = []
    current_sentence = []
    for word_item in words:
        if current_sentence and (
            word_item["start"] - current_sentence[-1]["end"] > sentence_gap_sec
        ):
            sentences.append(current_sentence)
            current_sentence = [word_item]
        else:
            current_sentence.append(word_item)
    if current_sentence:
        sentences.append(current_sentence)

    events = []
    for sentence_words in sentences:
        raw_text = "".join(word_item["word"] for word_item in sentence_words)
        if not raw_text.strip():
            continue

        # 关键纠错：让比对和最终通知文本均使用纠错后的规范汉字，以便能够正确匹配并上报“请求监护”等关键事件
        text = apply_corrections(raw_text)
        audio_timestamp = round(sentence_words[0]["start"], 2)

        # 设备码提取：用 finditer 拆分串联码，提取所有合法 9 字符设备码
        normalized_text = normalize_spoken_text(text)
        device_codes = [
            match.group(1) for match in NORM_DEVICE_PATTERN.finditer(normalized_text)
        ]

        # 收集所有匹配的关键字（严格拼音匹配，要求所有音节都出现）
        found_keywords = []
        for device_code in device_codes:
            found_keywords.append(device_code)
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
        for key_moment in found_keywords:
            if key_moment not in seen:
                seen.add(key_moment)
                unique_keywords.append(key_moment)

        # 每句推一条事件（带完整 text + key_moment 列表）
        # 如果有关键词/设备码，第一个事件带 text，后续事件只带 key_moment
        if unique_keywords:
            first = True
            for key_moment in unique_keywords:
                event = {
                    "localSec": audio_timestamp,
                    "key_moment": key_moment,
                    "keys": unique_keywords,
                }
                if first:
                    event["text"] = text
                    first = False
                events.append(event)
        else:
            # 没有关键词也推送完整文本（用于推理流展示）
            events.append({
                "localSec": audio_timestamp,
                "text": text,
                "key_moment": "",
                "keys": [],
            })

    return events
