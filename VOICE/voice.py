#!/usr/bin/env python3
"""
核电站操作视频语音转文本

技术方案:
  - Whisper large-v3 + 滑动窗口分段转录
  - noisereduce 稳态去噪
  - 140+条后处理纠错规则 (核电领域术语)
"""

import os, re, json, numpy as np, librosa, soundfile as sf
import noisereduce as nr
import whisper
from opencc import OpenCC
from tqdm import tqdm

# ======================== 路径配置 ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_AUDIO = os.path.join(BASE_DIR, "data", "_tmp_denoised.wav")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
SAMPLE_RATE = 16000
CC_T2S = OpenCC('t2s')

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

KEY_PHRASES = [
    "请求监护", "浮片对齐", "框架正确", "操作页数", "KICK",
    "T1RPA034", "1ES013V", "1ES001VB", "SM3", "SVideo", "地坑",
]


# ======================== 后处理函数 ========================

def apply_corrections(text: str) -> str:
    """应用后处理纠错规则，按最长匹配优先"""
    corrected = text
    for wrong, right in sorted(CORRECTIONS, key=lambda x: len(x[0]), reverse=True):
        corrected = corrected.replace(wrong, right)
    return re.sub(r'[。，]{3,}', '。', corrected)


# ======================== 音频处理 ========================

def denoise_nr_stationary(audio, sr):
    """noisereduce 稳态去噪"""
    return nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)


def _detect_speech_end(audio, sr, frame_len=2048, hop=512, rms_threshold=0.005):
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


def transcribe_segmented(audio, sr, model, prompt, window_sec=30, overlap_sec=5,
                         start_time=0, end_time=None, progress_cb=None,
                         segment_cb=None, progress_offset=0, progress_total=1):
    """
    滑动窗口分段转录 — 直接传 numpy 音频切片给 Whisper

    关键修复: 不再用 clip_timestamps + 文件路径方式调用 model.transcribe()，
    而是切片 numpy 数组后直接传入。这样 Whisper 只计算切片的 mel 频谱图，
    不存在全量加载 + seek 死循环的问题。

    Args:
        audio: np.ndarray, 去噪后的音频数组 (16kHz float32)
        sr: int, 采样率
        model: Whisper 模型实例
        prompt: str, initial_prompt
        segment_cb(new_words): 每个窗口转录后回调新增的词列表
        progress_cb(done_sec, total_sec): 进度回调
    """
    step = window_sec - overlap_sec
    if step <= 0:
        raise ValueError("window_sec 必须大于 overlap_sec")
    if end_time is None:
        end_time = len(audio) / sr

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
    pbar = tqdm(total=len(windows), desc="语音转录", unit="段")

    for i, (t, end) in enumerate(windows, start=1):
        if progress_cb:
            done_sec = progress_offset + max(0.0, t - start_time)
            progress_cb(done_sec, progress_total)

        # 切片音频
        start_sample = int(t * sr)
        end_sample = min(int(end * sr), len(audio))
        segment_audio = audio[start_sample:end_sample]

        # 跳过过短或静音段
        if len(segment_audio) < sr * 0.5:
            pbar.update(1)
            continue
        seg_rms = np.sqrt(np.mean(segment_audio ** 2))
        if seg_rms < 0.002:
            print(f"[Voice] 跳过静音段 {i}/{len(windows)} ({t:.1f}s-{end:.1f}s, RMS={seg_rms:.5f})")
            pbar.update(1)
            continue

        # 直接传 numpy 切片给 Whisper
        result = model.transcribe(segment_audio, language="zh", task="transcribe",
                                  verbose=False, word_timestamps=False,
                                  initial_prompt=prompt,
                                  condition_on_previous_text=False,
                                  temperature=0.0, beam_size=5,
                                  no_speech_threshold=0.6)

        seg_words = []
        for seg in result.get("segments", []):
            # Whisper 输出的时间戳是相对切片起始的，需要加上窗口起始时间 t
            st = round(t + float(seg.get("start", 0)), 3)
            et = round(t + float(seg.get("end", end - t)), 3)
            txt = CC_T2S.convert(seg.get("text", "").strip())
            if txt:
                seg_words.append({"word": txt, "start": st, "end": et})

        # 重叠区去重
        if i > 1 and all_words:
            mid = t + overlap_sec / 2
            seg_words = [w for w in seg_words if w["start"] >= mid]
            all_words = [w for w in all_words if w["start"] < mid]

        all_words.extend(seg_words)
        pbar.update(1)
        if segment_cb and seg_words:
            segment_cb(seg_words)
        if progress_cb:
            done_sec = progress_offset + max(0.0, end - start_time)
            progress_cb(done_sec, progress_total)

    pbar.close()
    return all_words


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


# ======================== 主函数 ========================

def main(progress_cb=None, segment_cb=None):
    """从视频音频中提取人说的话，转成文字。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载 + 去噪
    audio, sr = librosa.load(RAW_AUDIO, sr=SAMPLE_RATE)
    audio_duration = len(audio) / sr
    denoised = denoise_nr_stationary(audio, sr)

    # 裁剪尾部静音
    speech_end = _detect_speech_end(denoised, sr)
    if speech_end < audio_duration - 1.0:
        print(f"[Voice] 尾部静音: 语音结束于 {speech_end:.1f}s（总时长 {audio_duration:.1f}s）")
        effective_end = speech_end + 1.0
    else:
        effective_end = audio_duration

    # 加载 Whisper 模型
    model = whisper.load_model("large-v3")

    # 直接传 numpy 数组，不再写临时文件
    words = transcribe_segmented(denoised, sr, model, PROMPT_V4,
                                 window_sec=25, overlap_sec=5,
                                 start_time=0, end_time=effective_end,
                                 progress_cb=progress_cb,
                                 segment_cb=segment_cb,
                                 progress_offset=0,
                                 progress_total=effective_end)

    words = remove_hallucinations(words)

    full_text = "".join(w["word"] for w in words)
    full_text_corrected = apply_corrections(full_text)

    voice_events = build_voice_events(words)
    key_moments = build_key_moments(voice_events)

    with open(os.path.join(OUTPUT_DIR, "voice_events.json"), "w", encoding="utf-8") as f:
        json.dump(voice_events, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "key_moments.json"), "w", encoding="utf-8") as f:
        json.dump(key_moments, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "full_text.json"), "w", encoding="utf-8") as f:
        json.dump({"full_text": full_text_corrected, "word_count": len(words),
                    "duration_sec": audio_duration}, f, ensure_ascii=False, indent=2)


# ======================== 意图分类 + JSON 输出 ========================

DEVICE_PATTERN = re.compile(r'(1ES\w+|T1RPA\w+|LCO\w+|RPA\w+|SM3)')
ACTION_VERBS = ["开启", "关闭", "长按", "核对", "调出", "停运"]
CONFIRM_WORDS = ["好", "确认", "没问题", "收到", "明白", "正确"]


def classify_intent(text, prev_intent=""):
    """自动意图分类（基于关键词规则）"""
    if "请求监护" in text or "请监护" in text or "申请监护" in text:
        return "监护请求"
    if prev_intent == "监护请求" and any(w in text for w in ["好", "收到", "明白"]) and len(text) <= 6:
        return "监护确认"
    if "实验结束" in text or "试验结束" in text or "实验完成" in text or "试验完成" in text:
        return "实验结束"
    if "框架退出" in text:
        return "操作结束"
    has_device = bool(DEVICE_PATTERN.search(text))
    has_action = any(v in text for v in ACTION_VERBS)
    if has_device and has_action:
        return "操作指令"
    stripped = text.replace("好,", "").replace("好，", "").strip()
    if stripped in CONFIRM_WORDS or text.strip() in CONFIRM_WORDS:
        return "确认"
    if any(w in text for w in ["确认", "没问题"]):
        return "确认"
    if has_device:
        return "操作说明"
    return "其他"


def build_voice_events(words):
    """将 Whisper word 列表按停顿分句，生成 voice_events"""
    if not words:
        return []

    sentences, current = [], []
    for w in words:
        if current and w["start"] - current[-1]["end"] > 1.5:
            sentences.append(current)
            current = [w]
        else:
            current.append(w)
    if current:
        sentences.append(current)

    events = []
    prev_intent = ""
    for sent_words in sentences:
        text = "".join(w["word"] for w in sent_words)
        text = apply_corrections(text)
        if not text.strip():
            continue
        time_sec = round(sent_words[0]["start"], 2)
        intent = classify_intent(text, prev_intent)
        events.append({"time_sec": time_sec, "text": text, "intent": intent})
        prev_intent = intent

    return events


def build_key_moments(voice_events):
    """从 voice_events 提取关键时间点，供 MOT 模块使用"""
    moments = {
        "supervision_requests": [],
        "operation_commands": [],
        "operation_ends": [],
        "experiment_end": [],
    }
    for ev in voice_events:
        if ev["intent"] == "监护请求":
            moments["supervision_requests"].append({
                "timestamp_sec": ev["time_sec"], "text": ev["text"]
            })
        elif ev["intent"] == "操作指令":
            m = DEVICE_PATTERN.search(ev["text"])
            device = m.group(1) if m else ""
            moments["operation_commands"].append({
                "timestamp_sec": ev["time_sec"], "text": ev["text"], "device": device
            })
        elif ev["intent"] == "操作结束":
            moments["operation_ends"].append({
                "timestamp_sec": ev["time_sec"], "text": ev["text"]
            })
        elif ev["intent"] == "实验结束":
            moments["experiment_end"].append({
                "timestamp_sec": ev["time_sec"], "text": ev["text"]
            })
    return moments


if __name__ == "__main__":
    main()