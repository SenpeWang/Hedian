"""视觉流编码器: 把带标注的帧实时编码为 fragmented MP4 并推送到 Redis Stream.

每个视角(tracker=front / behavior=pop)持有一个持续 ffmpeg 子进程:
  - 视频轨:rawvideo 帧(BGR)经 stdin 喂入,PTS = 帧序号/fps,按推理节奏产出
  - 音频轨:front 从原视频文件复用(-i video_path -map 1:a);pop 无音频(-an)

输出 fMP4 字节流,按 MP4 box 边界切分为 init 段(ftyp+moov)与 media 段(moof+mdat),
xadd 到 inference:vis_stream:{view},供 VisStreamForwarder 消费.

注:标注已由各模块 draw_* 画进帧,本编码器只负责"帧→fMP4→Redis Stream",
不做任何可视化样式,不依赖 InferenceSync/global_sec 对齐(标注与帧物理一体).
"""
import logging
import struct
import subprocess
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger("core.vis_encoder")

KEY_PREFIX = "inference:vis_stream:"

_MOOF = b"moof"
_MDAT = b"mdat"


def _read_box(stream) -> Optional[tuple]:
    """从二进制流中读取一个完整 MP4 box.

    支持 64 位扩展 size 与 box 延伸到 EOF(size=0)两种特殊编码.

    Args:
        stream: 二进制流(如 subprocess stdout).

    Returns:
        (完整 box 字节, box 类型) 或 None(EOF / 截断).
    """
    header = stream.read(8)
    if len(header) < 8:
        return None
    size = struct.unpack(">I", header[:4])[0]
    box_type = header[4:8]
    if size == 1:
        # 64 位扩展 size
        ext = stream.read(8)
        if len(ext) < 8:
            return None
        size = struct.unpack(">Q", ext)[0]
        body_size = size - 16
        body = stream.read(body_size) if body_size > 0 else b""
        if body_size > 0 and len(body) < body_size:
            return None
        return header + ext + body, box_type
    if size == 0:
        # box 延伸到 EOF
        body = stream.read()
        return header + body, box_type
    body_size = size - 8
    body = stream.read(body_size) if body_size > 0 else b""
    if body_size > 0 and len(body) < body_size:
        return None
    return header + body, box_type


class VisEncoder:
    """单视角的 fMP4 编码器(持续 ffmpeg 进程 + box 切分)."""

    def __init__(
        self,
        view: str,
        video_path: Optional[str],
        fps: float,
        width: int = 0,
        height: int = 0,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        with_audio: bool = True,
    ):
        """初始化编码器.

        Args:
            view: 视角名, 用于拼接 Redis Stream key.
            video_path: 原视频路径; 非 None 且 with_audio 为真时复用其音频轨.
            fps: 输出帧率.
            width: 帧宽; 0 表示首帧时按实际 shape 确定.
            height: 帧高; 0 表示首帧时按实际 shape 确定.
            redis_host: Redis 主机地址.
            redis_port: Redis 端口.
            redis_db: 数据库编号.
            with_audio: 是否携带音频轨.
        """
        self.view = view
        self.video_path = video_path
        self.fps = fps
        self.width = int(width)
        self.height = int(height)
        # 只有提供原视频路径且显式需要音频时才复用音频轨
        self.with_audio = bool(with_audio and video_path)
        self._stream_key = KEY_PREFIX + view

        import redis
        # 二进制 payload,必须 decode_responses=False
        self._redis = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=False
        )
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._stderr_drain: Optional[threading.Thread] = None
        self._stopped = False
        self._init_sent = False
        self._frame_count = 0

    def _build_cmd(self) -> list:
        """拼装 ffmpeg 命令行.

        Returns:
            ffmpeg 命令行参数列表, stdin 为 rawvideo 输入, stdout 为 fMP4 输出.
        """
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
            "-i", "pipe:0",
        ]
        if self.with_audio:
            # 不用 -shortest:音频文件会被快速读完,shortest 令 ffmpeg 在音频 EOF 时立即退出不等视频
            cmd += ["-i", self.video_path, "-map", "0:v", "-map", "1:a"]
        else:
            cmd += ["-map", "0:v", "-an"]
        cmd += [
            "-c:v", "libx264", "-preset", "ultrafast",
            "-profile:v", "baseline", "-level", "3.1",
            "-pix_fmt", "yuv420p",
            "-g", str(max(1, int(self.fps))),  # 每 ~1s 一个关键帧
            "-f", "mp4",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "pipe:1",
        ]
        return cmd

    def start(self) -> None:
        """启动 ffmpeg 子进程与读取/排空线程."""
        if not self.width or not self.height:
            raise RuntimeError("VisEncoder 启动前需设置 width/height")
        cmd = self._build_cmd()
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_drain = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_drain.start()
        logger.info(
            f"VisEncoder[{self.view}] 启动 ffmpeg: "
            f"{self.width}x{self.height}@{self.fps}fps audio={self.with_audio}"
        )

    def feed_frame(self, frame: np.ndarray) -> None:
        """喂入一帧(已带标注),惰性启动编码器(首帧时据 shape 确定分辨率).

        编码失败不应拖垮推理:写失败时置 _stopped,推理继续但该视角无流.

        Args:
            frame: BGR 帧, 形状 (H, W, 3).
        """
        if self._stopped:
            return
        if self._proc is None:
            frame_height, frame_width = frame.shape[:2]
            self.height, self.width = int(frame_height), int(frame_width)
            self.start()
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            self._frame_count += 1
        except (BrokenPipeError, OSError) as e:
            logger.error(f"VisEncoder[{self.view}] 写帧失败: {e}")
            self._stopped = True

    def _read_loop(self) -> None:
        """从 ffmpeg stdout 解析 MP4 box,切分 init/media 段写 Redis Stream."""
        stream = self._proc.stdout
        init_buf = []
        pending_moof = None
        try:
            while not self._stopped:
                box = _read_box(stream)
                if box is None:
                    break
                data, box_type = box
                if box_type == _MOOF:
                    # 首次遇到 moof → 之前累积的 ftyp+moov 即 init 段
                    if init_buf and not self._init_sent:
                        self._xadd("init", b"".join(init_buf))
                        self._init_sent = True
                        init_buf = []
                    pending_moof = data
                elif box_type == _MDAT:
                    if pending_moof is not None:
                        self._xadd("media", pending_moof + data)
                        pending_moof = None
                else:
                    # ftyp / moov 等头部 box → 累积为 init
                    if not self._init_sent:
                        init_buf.append(data)
            # 进程结束前 flush 残留 init
            if init_buf and not self._init_sent:
                self._xadd("init", b"".join(init_buf))
                self._init_sent = True
        except Exception as e:
            logger.error(f"VisEncoder[{self.view}] 读取循环异常: {e}")
        finally:
            # end 段必须等 ffmpeg 真正退出后才写入(有界等待):
            # reader 提前退出而 ffmpeg 仍在产出时立即写 end, 会令转发器误判供给终止
            return_code = None
            for _ in range(15):
                try:
                    return_code = self._proc.wait(timeout=1)
                    break
                except Exception:
                    continue
            if return_code is not None:
                logger.info(f"VisEncoder[{self.view}] ffmpeg 退出 rc={return_code}, 共喂 {self._frame_count} 帧")
            else:
                logger.warning(f"VisEncoder[{self.view}] ffmpeg 15s 未退出, 强制收尾写 end")
            self._xadd("end", b"")

    def _xadd(self, segment_type: str, data: bytes) -> None:
        """向视觉流 Redis Stream 写入一个段.

        Args:
            segment_type: 段类型, 取值 "init" / "media" / "end".
            data: 段字节内容; "end" 段为空字节.
        """
        try:
            self._redis.xadd(
                self._stream_key,
                {"type": segment_type.encode(), "data": data},
                maxlen=200000,
                approximate=True,
            )
        except Exception as e:
            logger.error(f"VisEncoder[{self.view}] xadd {segment_type} 失败: {e}")

    def _drain_stderr(self) -> None:
        """排空 ffmpeg stderr(防管道满阻塞),记录非进度行用于诊断."""
        try:
            for raw in iter(self._proc.stderr.readline, b""):
                if self._stopped:
                    break
                line = raw.decode("utf-8", "ignore").strip()
                if line and not any(k in line for k in ("frame=", "fps=", "size=", "bitrate=", "speed=", "cpb:", "Side data")):
                    logger.warning(f"ffmpeg[{self.view}]: {line}")
        except Exception:
            pass

    def finalize(self) -> None:
        """收尾: 先关 stdin 让 ffmpeg flush 剩余段, 由读取线程读到 EOF 自然收尾后再终止.

        严禁在关闭 stdin 前置 _stopped: reader 线程的 while 条件会立即退出,
        ffmpeg 尚未 flush 的最后几段 media 永远留在管道里无人转发(前端尾部断档).
        """
        try:
            if self._proc and self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        if self._reader:
            self._reader.join(timeout=15)
        if self._reader and self._reader.is_alive():
            # reader 卡住(ffmpeg 未产出 EOF)才强制杀进程逼出 end
            logger.warning(f"VisEncoder[{self.view}] 读取线程 15s 未结束, 强制终止 ffmpeg")
            try:
                self._proc.kill()
            except Exception:
                pass
            self._reader.join(timeout=5)
        self._stopped = True   # 此后 feed_frame 幂等短路(reader 已完成收尾)
        if self._proc:
            try:
                self._proc.wait(timeout=15)
            except Exception:
                pass
        logger.info(f"VisEncoder[{self.view}] 结束,共喂 {self._frame_count} 帧")
