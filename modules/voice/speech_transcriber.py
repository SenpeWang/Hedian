"""
语音转文字模块

使用 Qwen3-ASR 进行语音转录，带后处理纠错和幻觉移除。
技术方案来自备份版本 A_DemoSrc1/VOICE/voice.py
"""
import os
import re
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger("module.voice.transcriber")



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
    ("请结束", "请求监护"), ("请求进入", "请求监护"), ("请求接入", "请求监护"),
    ("请台军务", "请求监护"), ("请台监护", "请求监护"),
    ("台军务", "请求监护"),

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
    ("手放", "收到"), ("三黑", "三K"), ("SV六十", "SVideo"),

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
    """移除连续重复的短词（ASR 幻觉）"""
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
                        st = round(t + float(seg.get("start", 0.0)), 3)
                        et = round(t + float(seg.get("end", 0.0)), 3)
                        seg_item = {"word": txt, "start": st, "end": et}
                        if "words" in seg and seg["words"]:
                            seg_item["words"] = [
                                {
                                    "start": round(t + float(w.get("start", 0.0)), 3),
                                    "end": round(t + float(w.get("end", 0.0)), 3),
                                    "word": w.get("word") or w.get("text") or "",
                                }
                                for w in seg["words"]
                            ]
                        seg_words.append(seg_item)

                # midpoint去重: 只取当前窗口新增的后半段
                if overlap_sec > 0 and i > 1 and all_words:
                    mid = t + overlap_sec / 2
                    # 只保留当前窗口 > mid 的内容
                    new_words = []
                    for w in seg_words:
                        if w["start"] >= mid:
                            new_words.append(w)
                        elif "words" in w and w["words"]:
                            kept = [wd for wd in w["words"] if wd["start"] >= mid]
                            if kept:
                                new_words.append({
                                    "word": "".join(wd["word"] for wd in kept),
                                    "start": kept[0]["start"],
                                    "end": w["end"],
                                    "words": kept,
                                })
                    seg_words = new_words

                all_words.extend(seg_words)

                if progress_callback:
                    progress_callback(cur_end, end_time)

                if seg_words and on_segment:
                    corrected = []
                    for w in seg_words:
                        cw = dict(w)
                        cw["word"] = apply_corrections(cw["word"])
                        corrected.append(cw)
                    on_segment(corrected, t, cur_end, end_time)

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


# ======================== 文本归一化与关键词提取 (原 intent_classifier) ========================

NORM_DEVICE_PATTERN = re.compile(r"([1-9]?[A-Z]{2,3}\d{3}[A-Z]{2}|1EAS\w+|T\d*RPA\w+|LCO[\w\.]+|RPA\w+|SM3)")

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
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).upper()
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
    text = text.replace("EES", "1EAS").replace("EAS", "1EAS").replace("EEX", "1EAS")
    text = text.replace("ES", "1EAS")
    # 只保留字母数字和小数点
    return re.sub(r"[^A-Z0-9\.]", "", text)

def get_key_moment(text: str, device: str) -> str:
    """根据匹配到的语音关键字返回对应的 key_moment"""
    if device:
        return device
    if "监护" in text:
        return "请求监护"
    if "执行" in text:
        return "执行"
    if "核对" in text or "核实" in text:
        return "核对"
    if "信息通报" in text:
        return "信息通报"
    if "信息通告" in text:
        return "信息通告"
    if "通报完毕" in text:
        return "通报完毕"
    if "通告完毕" in text:
        return "通告完毕"
    if text.strip() == "收到":
        return "收到"
    return ""

def process_transcribed_words(words: List[Dict], sentence_gap_sec: float = 1.0) -> List[Dict]:
    """对字/词按停顿切分出段落，并提取关键事件"""
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
        text = "".join(w["word"] for w in sent_words)
        if not text.strip():
            continue

        audio_ts = round(sent_words[0]["start"], 2)

        # 设备码提取
        norm_text = normalize_spoken_text(text)
        m = NORM_DEVICE_PATTERN.search(norm_text)
        device = m.group(1) if m else ""

        # 收集所有匹配的关键字
        found_keywords = set()
        if device:
            found_keywords.add(device)
        for kw, label in [("监护", "请求监护"), ("执行", "执行"), ("核对", "核对"), ("核实", "核对"),
                          ("信息通报", "信息通报"), ("信息通告", "信息通告"),
                          ("通报完毕", "通报完毕"), ("通告完毕", "通告完毕")]:
            if kw in text:
                found_keywords.add(label)
        if text.strip() == "收到":
            found_keywords.add("收到")

        # 每句先推完整文本（仅第一个事件带 text）
        first = True
        for km in found_keywords:
            ev = {"localSec": audio_ts, "key_moment": km}
            if first:
                ev["text"] = text
                first = False
            events.append(ev)

        # 如果没有匹配任何关键字，也推文本
        if not found_keywords:
            events.append({"localSec": audio_ts, "text": text, "key_moment": ""})

    return events
