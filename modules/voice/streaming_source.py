"""流式音频源：ffmpeg 管道 + 增量 PCM 块.

通过 ffmpeg stdout 管道增量产出 16kHz 单声道 float32 采样块：

  - 内存占用与音频总时长无关（配合 `PcmRing` 有界保留）；
  - 本地文件与 rtsp/rtmp/http 直播地址走同一条代码路径；
  - 不产生任何临时 wav 文件。

配合 `modules/voice/segmenters.py` 的在线切片，即可实现"读到一段语音就解一段"
的流式解码。
"""

from __future__ import annotations

import logging
import subprocess
from typing import Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("module.voice.stream")

# 统一工作采样率（与模型、VAD 参数一致）
SAMPLE_RATE = 16000
# 每次从管道读取的音频时长（秒）：越小首包延迟越低、系统调用越多
DEFAULT_BLOCK_SEC = 0.2
# int16 → float32 归一化系数
INT16_SCALE = 32768.0

# ffmpeg 视为"无 EOF 直播源"的协议前缀
_LIVE_SCHEMES = (
    "rtsp://",
    "rtmps://",
    "rtmp://",
    "http://",
    "https://",
    "udp://",
    "tcp://",
    "srt://",
)


def is_live_source(source: str) -> bool:
    """判断输入是否为无 EOF 语义的实时流地址.

    Args:
        source: 输入路径或 URL.

    Returns:
        实时流返回 True；本地文件返回 False.
    """
    lowered = str(source).strip().lower()
    return lowered.startswith(_LIVE_SCHEMES)


class PcmStream:
    """ffmpeg 管道驱动的 16kHz 单声道 PCM 流（可迭代，逐块产出 float32 采样）.

    典型用法::

        with PcmStream("cam.mp4") as stream:
            for pcm in stream:
                ...  # pcm: np.ndarray[float32], 长度约为 block_sec * 16000
    """

    def __init__(
        self,
        source: str,
        sample_rate: int = SAMPLE_RATE,
        block_sec: float = DEFAULT_BLOCK_SEC,
        ffmpeg_bin: str = "ffmpeg",
        extra_input_args: Optional[List[str]] = None,
    ) -> None:
        """初始化流式音频源（不启动 ffmpeg）.

        Args:
            source: 输入路径或 URL（文件 / rtsp / rtmp / http）.
            sample_rate: 输出采样率（Hz）.
            block_sec: 单次读取块时长（秒），决定首包延迟粒度.
            ffmpeg_bin: ffmpeg 可执行文件路径.
            extra_input_args: 追加在 `-i` 之前的 ffmpeg 输入参数.
        """
        self.source = str(source)
        self.sample_rate = int(sample_rate)
        self.block_sec = float(block_sec)
        self.ffmpeg_bin = str(ffmpeg_bin)
        self._extra_input_args = list(extra_input_args or [])
        self._proc: Optional[subprocess.Popen] = None
        # 单块目标采样数；实际读取以字节数为准，末尾块可能更短
        self._block_samples = max(
            1, int(round(self.block_sec * self.sample_rate))
        )
        self.live = is_live_source(self.source)

    # ── 命令与环境 ──────────────────────────────────────────────

    def _build_command(self) -> List[str]:
        """组装 ffmpeg 命令：解码任意输入 → 16k 单声道 s16le → stdout.

        Returns:
            ffmpeg 参数列表.
        """
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
        ]
        if self.live:
            # 直播源：优先 TCP（丢包可见、可重连），并设置读取超时避免永久挂起
            cmd += ["-rtsp_transport", "tcp", "-rw_timeout", "15000000"]
        cmd += self._extra_input_args
        cmd += [
            "-i",
            self.source,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "s16le",
            "-",
        ]
        return cmd

    # ── 生命周期 ────────────────────────────────────────────────

    def open(self) -> None:
        """启动 ffmpeg 子进程（幂等）."""
        if self._proc is not None:
            return
        command = self._build_command()
        logger.info("启动流式音频源: %s (live=%s)", self.source, self.live)
        self._proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def close(self) -> None:
        """关闭管道并回收 ffmpeg 子进程.

        先关闭 stdout 让 ffmpeg 收到 EPIPE 自然退出，超时后再强杀，避免僵尸进程。
        """
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg 未在 5s 内退出，强制终止")
            proc.kill()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                pass
        try:
            if proc.stderr:
                tail = proc.stderr.read() or b""
                proc.stderr.close()
                if tail:
                    message = tail.decode("utf-8", errors="ignore")[-500:]
                    # 主动停止（提前关闭 stdout）会让 ffmpeg 报 Broken pipe，属正常路径
                    if "Broken pipe" in message:
                        logger.debug("ffmpeg 因管道关闭退出: %s", message)
                    else:
                        logger.warning("ffmpeg stderr: %s", message)
        except Exception:
            pass

    def __enter__(self) -> "PcmStream":
        """进入 with 块: 打开音频流并返回自身.

        Returns:
            自身, 供 with 语句绑定.
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """退出 with 块: 关闭音频流.

        Args:
            exc_type: 异常类型; 正常退出时为 None.
            exc: 异常实例; 正常退出时为 None.
            tb: 异常回溯对象; 正常退出时为 None.
        """
        self.close()

    # ── 读取 ────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[np.ndarray]:
        """逐块产出 float32 PCM（范围 [-1, 1]）.

        Yields:
            一维 float32 采样数组；长度约为 block_sec * sample_rate.
        """
        if self._proc is None:
            self.open()
        assert self._proc is not None and self._proc.stdout is not None
        read_bytes = self._block_samples * 2  # int16 = 2 字节/采样
        stdout = self._proc.stdout
        while True:
            raw = stdout.read(read_bytes)
            if not raw:
                break
            if len(raw) % 2:  # 丢弃半个采样，避免 frombuffer 报错
                raw = raw[:-1]
            if not raw:
                continue
            pcm = (
                np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                / INT16_SCALE
            )
            yield pcm


class PcmRing:
    """有界 PCM 环形缓冲：按"绝对秒"切片，自动丢弃已消费前缀.

    在线切片是"事后确认"的：一段语音要等到尾部静音达到阈值才被判定结束，
    此时该段的起点可能已在 `max_chunk` 秒之前。因此必须把最近一段音频保留在
    内存里，等段闭合后再回取。

    保留窗口必须覆盖::

        max_chunk + min_silence + 2 * speech_pad + boundary_overlap

    超出部分按整块丢弃，保证内存与音频总时长无关。
    """

    def __init__(
        self, retain_sec: float, sample_rate: int = SAMPLE_RATE
    ) -> None:
        """初始化环形缓冲.

        Args:
            retain_sec: 需要保留的历史音频时长（秒）.
            sample_rate: 采样率（Hz）.
        """
        self._retain = max(1.0, float(retain_sec))
        self._sr = int(sample_rate)
        self._chunks: List[Tuple[float, np.ndarray]] = []
        self._start_sec: Optional[float] = None
        self._end_sec: float = 0.0

    def append(
        self, pcm: np.ndarray, start_sec: Optional[float] = None
    ) -> None:
        """追加一块 PCM，并按保留窗口裁剪历史.

        Args:
            pcm: 一维 PCM 采样.
            start_sec: 该块首样本的绝对时间（秒）；None 时紧接当前末尾.
        """
        if pcm is None or len(pcm) == 0:
            return
        if start_sec is None:
            start_sec = self._end_sec
        start_sec = float(start_sec)
        if self._start_sec is None:
            self._start_sec = start_sec
        self._chunks.append((start_sec, np.asarray(pcm, dtype=np.float32)))
        self._end_sec = start_sec + len(pcm) / self._sr
        self._trim()

    def _trim(self) -> None:
        """丢弃整个已滑出保留窗口的前缀块."""
        keep_from = self._end_sec - self._retain
        while len(self._chunks) > 1:
            chunk_start, chunk = self._chunks[0]
            chunk_end = chunk_start + len(chunk) / self._sr
            if chunk_end <= keep_from:
                self._chunks.pop(0)
                self._start_sec = self._chunks[0][0]
            else:
                break

    def slice(self, start_sec: float, end_sec: float) -> np.ndarray:
        """按绝对秒区间切片，越界部分自动截断.

        Args:
            start_sec: 区间起始绝对时间（秒）.
            end_sec: 区间结束绝对时间（秒）.

        Returns:
            一维 float32 采样数组；区间无有效数据时返回空数组.
        """
        pieces: List[np.ndarray] = []
        for chunk_start, chunk in self._chunks:
            chunk_end = chunk_start + len(chunk) / self._sr
            if chunk_end <= start_sec or chunk_start >= end_sec:
                continue
            # 换算到块内样本下标，四舍五入保证相邻段拼接无累积漂移
            local_start = int(
                round((max(start_sec, chunk_start) - chunk_start) * self._sr)
            )
            local_end = int(
                round((min(end_sec, chunk_end) - chunk_start) * self._sr)
            )
            local_start = max(0, min(local_start, len(chunk)))
            local_end = max(0, min(local_end, len(chunk)))
            if local_end > local_start:
                pieces.append(chunk[local_start:local_end])
        if not pieces:
            return np.zeros((0,), dtype=np.float32)
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces)

    @property
    def start_sec(self) -> float:
        """缓冲中最早样本的绝对时间（秒），空时为 0.

        Returns:
            缓冲中最早样本的绝对时间(秒); 缓冲为空时为 0.0.
        """
        return float(self._start_sec or 0.0)

    @property
    def end_sec(self) -> float:
        """缓冲中最新样本之后的绝对时间（秒）.

        Returns:
            缓冲中最新样本之后的绝对时间(秒).
        """
        return float(self._end_sec)

    def __len__(self) -> int:
        """缓冲内保留的采样数.

        Returns:
            缓冲内保留的采样数.
        """
        return sum(len(chunk) for _, chunk in self._chunks)
