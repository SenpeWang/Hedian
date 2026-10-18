"""可插拔语音切片策略：把"整文件静音扫描"改为"在线逐帧端点检测".

原实现（`SpeechTranscriber._detect_speech_intervals`）先跑一遍
`ffmpeg -af silencedetect` 扫完整段音频、解析 stderr 拿到全部静音区间，再逐段
切 wav 解码。该方式有三个固有问题：

  1. **必须等文件读完**：静音区间要读到 EOF 才能闭合，无法流式、无法接直播；
  2. **切点不可控**：`max_chunk` 超长时按固定 30s 硬切，可能切在字中间
     （实测 "收到" 被切成 "收" / "到"）；
  3. **策略不可换**：想试别的切法（滑窗 / 更短静音 / 不同阈值）只能改代码。

本模块把切片抽象成策略接口，运行时由 `config.yaml → voice.segmenter` 选择：

  - ``silence_vad``：在线能量 VAD，静音达到阈值即封段；长语音强制切分时
    会在最近窗口内**挑能量最低点**作为切点（词边界保护），不再盲切 30s；
  - ``fixed_window``：固定窗 + 重叠（旧版 `Hedian` 的滑窗逻辑），用于 A/B 对比。

新增策略只需继承 `BaseSegmenter` 并用 `@register_segmenter` 装饰即可，便于
持续试验"哪种切片转录效果更好"。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np

logger = logging.getLogger("module.voice.segmenter")

SAMPLE_RATE = 16000
# 能量判决的帧长（秒）：20ms 是语音端点检测的常规粒度
FRAME_SEC = 0.02


@dataclass
class Segment:
    """一个待解码的语音区间（绝对时间，秒）."""

    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        """区间时长（秒）.

        Returns:
            区间时长(秒).
        """
        return self.end - self.start


class BaseSegmenter(ABC):
    """切片策略基类：接收 PCM 块，产出闭合的语音区间.

    子类只需实现 `_consume_frame`（逐 20ms 帧判决）与 `_finish`（收尾），
    基类负责帧缓冲、绝对时间对齐与下标编号。
    """

    name: ClassVar[str] = "base"

    def __init__(self, sample_rate: int = SAMPLE_RATE, **config: Any) -> None:
        """初始化帧缓冲.

        Args:
            sample_rate: 采样率（Hz）.
            **config: 策略参数（由子类解释，未识别的键忽略）.
        """
        self.sample_rate = int(sample_rate)
        self._frame_len = max(1, int(round(FRAME_SEC * self.sample_rate)))
        self._frame_dur = self._frame_len / self.sample_rate
        # 未凑满一帧的残留采样，及其在流中的绝对起始时间
        self._pending = np.zeros((0,), dtype=np.float32)
        self._pending_start = 0.0
        self._index = 0
        self._config = dict(config)

    # ── 对外接口 ────────────────────────────────────────────────

    def push(
        self, pcm: np.ndarray, start_sec: Optional[float] = None
    ) -> List[Segment]:
        """喂入一块 PCM，返回本次判定出的、已经闭合的语音区间.

        Args:
            pcm: 一维 PCM 采样（float32，范围 [-1, 1]）.
            start_sec: 该块首样本的绝对时间（秒）；None 表示紧接上次末尾.

        Returns:
            已闭合的区间列表（按时间升序）；通常为 0 或 1 个.
        """
        if pcm is None or len(pcm) == 0:
            return []
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if start_sec is None:
            start_sec = (
                self._pending_start + len(self._pending) / self.sample_rate
            )
        start_sec = float(start_sec)

        if self._pending.size == 0:
            self._pending = pcm
            self._pending_start = start_sec
        else:
            self._pending = np.concatenate([self._pending, pcm])

        segments: List[Segment] = []
        while self._pending.size >= self._frame_len:
            frame = self._pending[: self._frame_len]
            frame_start = self._pending_start
            segments.extend(self._consume_frame(frame, frame_start))
            frame_len = self._frame_len
            self._pending = self._pending[frame_len:]
            self._pending_start = frame_start + self._frame_dur
        return segments

    def flush(self) -> List[Segment]:
        """流结束时收尾：处理残留采样并封闭仍在进行中的区间.

        Returns:
            收尾产生的区间列表.
        """
        segments: List[Segment] = []
        if self._pending.size > 0:
            segments.extend(
                self._consume_frame(self._pending, self._pending_start)
            )
            self._pending = np.zeros((0,), dtype=np.float32)
        segments.extend(self._finish())
        return segments

    def retain_sec(self) -> float:
        """环形缓冲需要保留的历史音频长度（秒）.

        在线切片是"事后确认"的：段闭合时其起点可能已在很久之前，因此环形缓冲
        至少要覆盖子类可能回看的最大跨度。

        Returns:
            保留时长（秒）.
        """
        return 60.0

    def describe(self) -> Dict[str, Any]:
        """返回策略名与生效参数，用于日志排查与 A/B 对比.

        Returns:
            含 name 与参数的字典.
        """
        return {"name": self.name, **self._config}

    # ── 子类实现 ────────────────────────────────────────────────

    @abstractmethod
    def _consume_frame(
        self, frame: np.ndarray, frame_start: float
    ) -> List[Segment]:
        """处理单个 20ms 帧.

        Args:
            frame: 当前帧采样.
            frame_start: 帧起始绝对时间（秒）.

        Returns:
            该帧触发的闭合区间列表.
        """
        raise NotImplementedError

    def _finish(self) -> List[Segment]:
        """流结束时的收尾钩子，默认无动作.

        Returns:
            收尾产生的区间列表.
        """
        return []

    def _new_segment(self, start: float, end: float) -> Segment:
        """构造并编号一个区间（统一 3 位小数，避免浮点噪声）.

        Args:
            start: 区间起点（秒）.
            end: 区间终点（秒）.

        Returns:
            编号后的区间.
        """
        segment = Segment(
            index=self._index,
            start=round(float(start), 3),
            end=round(float(end), 3),
        )
        self._index += 1
        return segment


class SilenceVadSegmenter(BaseSegmenter):
    """在线能量 VAD 切片：静音达到阈值即封段.

    与整文件 `silencedetect` 的语义对齐（同样用 dB 阈值 + 最短静音），但判决是
    逐帧在线进行的，因此不需要等文件结束。长语音超过 `max_chunk` 时，不按固定
    时长硬切，而是在最近 `split_search_sec` 窗口内挑**能量最低点**切分，尽量
    避免把词切成两半。
    """

    name = "silence_vad"

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        noise_db: float = -35.0,
        min_silence: float = 0.8,
        speech_pad: float = 0.2,
        min_speech: float = 0.35,
        max_chunk: float = 30.0,
        split_search_sec: float = 1.0,
        split_margin_db: float = 6.0,
        **config: Any,
    ) -> None:
        """初始化在线 VAD 切片器.

        Args:
            sample_rate: 采样率（Hz）.
            noise_db: 语音/静音判决阈值（dB）.
            min_silence: 判为段结束所需的最短静音时长（秒）.
            speech_pad: 每段前后保留的余量（秒）.
            min_speech: 丢弃短于该时长的段（秒），按加余量前的纯语音跨度判定.
            max_chunk: 单段最长时长（秒），超过则强制切分.
            split_search_sec: 强制切分时回看并寻找最低能量点的窗口（秒）.
            split_margin_db: 强制切分时"接近最低能量"的容差（dB）；从窗口末尾
                向前找第一个落入该容差的帧，既避开字中间又尽量少截短段落.
            **config: 透传给基类.
        """
        super().__init__(sample_rate=sample_rate, **config)
        self.noise_db = float(noise_db)
        self.min_silence = float(min_silence)
        self.speech_pad = float(speech_pad)
        self.min_speech = float(min_speech)
        self.max_chunk = float(max_chunk)
        self.split_search_sec = max(FRAME_SEC, float(split_search_sec))
        self.split_margin_db = max(0.0, float(split_margin_db))
        self._in_speech = False
        self._speech_start = 0.0
        self._last_voice_end = 0.0
        self._silence_run = 0.0
        # 最近若干帧的 (帧结束时间, dB)，用于强制切分时找最低能量点
        self._recent: deque = deque()

    def _consume_frame(
        self, frame: np.ndarray, frame_start: float
    ) -> List[Segment]:
        """在线判决单帧：维护语音/静音状态机并产出闭合区间.

        Args:
            frame: 当前帧采样.
            frame_start: 帧起始绝对时间（秒）.

        Returns:
            该帧触发的闭合区间列表.
        """
        frame_dur = len(frame) / self.sample_rate
        frame_end = frame_start + frame_dur
        rms = float(np.sqrt(np.mean(frame**2))) if frame.size else 0.0
        db = 20.0 * float(np.log10(rms + 1e-10))
        is_voice = db > self.noise_db

        self._recent.append((frame_end, db))
        cutoff = frame_end - self.split_search_sec
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()

        segments: List[Segment] = []

        if is_voice:
            if not self._in_speech:
                self._in_speech = True
                self._speech_start = frame_start
                self._recent.clear()
                self._recent.append((frame_end, db))
            self._silence_run = 0.0
            self._last_voice_end = frame_end
        elif self._in_speech:
            self._silence_run += frame_dur
            if self._silence_run >= self.min_silence:
                segment = self._emit(self._speech_start, self._last_voice_end)
                if segment is not None:
                    segments.append(segment)
                self._in_speech = False
                self._silence_run = 0.0

        # 长语音保护：超过 max_chunk 在最低能量点切分，避免硬切在字中间
        if (
            self._in_speech
            and (frame_end - self._speech_start) >= self.max_chunk
        ):
            cut = self._pick_split_point(self._speech_start + self.max_chunk)
            segment = self._emit(self._speech_start, cut)
            if segment is not None:
                segments.append(segment)
            self._speech_start = cut
            self._silence_run = 0.0
            self._recent.clear()
            self._recent.append((frame_end, db))

        return segments

    def _pick_split_point(self, target: float) -> float:
        """在 [target - split_search_sec, target] 内挑能量最低点作为切点.

        从窗口末尾向前扫描，取第一个"落入最低能量 + split_margin_db 容差"的帧：
        既保证切在低能量处（大概率是字间/词间空隙，而非字中间），又避免为了
        追求全局最低点而把本段截得过短。

        Args:
            target: 期望切分时间（秒），即 speech_start + max_chunk.

        Returns:
            实际切分时间（秒）；无可用帧时返回 target.
        """
        if not self._recent:
            return target
        lower = target - self.split_search_sec
        window = [
            (frame_end, db)
            for frame_end, db in self._recent
            if lower <= frame_end <= target + self._frame_dur
        ]
        if not window:
            return target
        min_db = min(db for _, db in window)
        threshold = min_db + self.split_margin_db
        for frame_end, db in reversed(window):
            if db <= threshold:
                return float(frame_end)
        return target

    def _emit(self, start: float, end: float) -> Optional[Segment]:
        """按 speech_pad / min_speech 约束产出一个区间.

        min_speech 按**加余量前**的纯语音跨度判定（与整文件 silencedetect 语义
        一致），避免 speech_pad 把一段退化的零长语音"撑"成合法区间。

        Args:
            start: 语音起点（秒）.
            end: 语音终点（秒）.

        Returns:
            区间；纯语音跨度短于 min_speech 时返回 None.
        """
        if end - start < self.min_speech:
            return None
        padded_start = max(0.0, start - self.speech_pad)
        padded_end = end + self.speech_pad
        return self._new_segment(padded_start, padded_end)

    def _finish(self) -> List[Segment]:
        """收尾：封闭仍在进行中的语音段.

        Returns:
            收尾产生的区间列表.
        """
        if not self._in_speech:
            return []
        self._in_speech = False
        segment = self._emit(self._speech_start, self._last_voice_end)
        return [segment] if segment is not None else []

    def retain_sec(self) -> float:
        """保留窗口：覆盖 max_chunk 回看 + 静音判定 + 前后余量.

        Returns:
            保留时长（秒）.
        """
        return self.max_chunk + self.min_silence + 2 * self.speech_pad + 5.0

    def describe(self) -> Dict[str, Any]:
        """返回策略名与全部生效参数，用于日志排查与 A/B 对比.

        Returns:
            参数字典.
        """
        return {
            "name": self.name,
            "noise_db": self.noise_db,
            "min_silence": self.min_silence,
            "speech_pad": self.speech_pad,
            "min_speech": self.min_speech,
            "max_chunk": self.max_chunk,
            "split_search_sec": self.split_search_sec,
            "split_margin_db": self.split_margin_db,
        }


class FixedWindowSegmenter(BaseSegmenter):
    """固定窗 + 重叠切片（旧版 `Hedian` 的滑窗逻辑）.

    仅用于 A/B 对比：不依赖静音检测，按固定步长均匀切分。切点与语音内容无关，
    因此更容易切在字中间；配合 `boundary_overlap_sec` 与跨段句子组装可缓解。
    """

    name = "fixed_window"

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_sec: float = 20.0,
        overlap_sec: float = 5.0,
        min_speech: float = 0.35,
        **config: Any,
    ) -> None:
        """初始化固定窗切片器.

        Args:
            sample_rate: 采样率（Hz）.
            window_sec: 窗长（秒）.
            overlap_sec: 相邻窗重叠时长（秒）.
            min_speech: 末尾残窗短于该时长时丢弃（秒）.
            **config: 透传给基类.
        """
        super().__init__(sample_rate=sample_rate, **config)
        self.window_sec = float(window_sec)
        self.overlap_sec = float(overlap_sec)
        self.min_speech = float(min_speech)
        self._step = self.window_sec - self.overlap_sec
        if self._step <= 0:
            self._step = self.window_sec
        self._next_start = 0.0
        self._cursor_end = 0.0

    def _consume_frame(
        self, frame: np.ndarray, frame_start: float
    ) -> List[Segment]:
        """推进游标并在窗口填满时产出区间.

        Args:
            frame: 当前帧采样.
            frame_start: 帧起始绝对时间（秒）.

        Returns:
            该帧触发的闭合区间列表.
        """
        self._cursor_end = frame_start + len(frame) / self.sample_rate
        segments: List[Segment] = []
        while self._next_start + self.window_sec <= self._cursor_end:
            segments.append(
                self._new_segment(
                    self._next_start, self._next_start + self.window_sec
                )
            )
            self._next_start += self._step
        return segments

    def _finish(self) -> List[Segment]:
        """收尾：产出最后一个不足整窗但足够长的残窗.

        Returns:
            收尾产生的区间列表.
        """
        if self._cursor_end - self._next_start >= self.min_speech:
            segment = self._new_segment(self._next_start, self._cursor_end)
            return [segment]
        return []

    def retain_sec(self) -> float:
        """保留窗口：覆盖一个完整窗.

        Returns:
            保留时长（秒）.
        """
        return self.window_sec + 5.0

    def describe(self) -> Dict[str, Any]:
        """返回策略名与全部生效参数，用于日志排查与 A/B 对比.

        Returns:
            参数字典.
        """
        return {
            "name": self.name,
            "window_sec": self.window_sec,
            "overlap_sec": self.overlap_sec,
            "min_speech": self.min_speech,
        }


# ── 策略注册表 ──────────────────────────────────────────────────

_SEGMENTER_REGISTRY: Dict[str, Type[BaseSegmenter]] = {}


def register_segmenter(cls: Type[BaseSegmenter]) -> Type[BaseSegmenter]:
    """把切片策略注册进全局表.

    Args:
        cls: 继承 `BaseSegmenter` 的策略类.

    Returns:
        原类（便于作装饰器使用）.
    """
    _SEGMENTER_REGISTRY[cls.name] = cls
    return cls


def available_segmenters() -> List[str]:
    """列出所有已注册的切片策略名.

    Returns:
        策略名列表（按字母序）.
    """
    return sorted(_SEGMENTER_REGISTRY)


def build_segmenter(
    name: str, sample_rate: int = SAMPLE_RATE, **config: Any
) -> BaseSegmenter:
    """按名字构造切片策略实例.

    Args:
        name: 策略名（见 `available_segmenters`）.
        sample_rate: 采样率（Hz）.
        **config: 策略参数.

    Returns:
        切片策略实例.

    Raises:
        ValueError: 策略名未注册.
    """
    key = str(name or "").strip()
    if key not in _SEGMENTER_REGISTRY:
        raise ValueError(
            f"未知切片策略 '{name}'，可选: {available_segmenters()}"
        )
    return _SEGMENTER_REGISTRY[key](sample_rate=sample_rate, **config)


register_segmenter(SilenceVadSegmenter)
register_segmenter(FixedWindowSegmenter)
