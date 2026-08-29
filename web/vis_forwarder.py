"""视觉流转发器 — Web 进程后台线程,消费 Redis Stream 中的 fMP4 段,
经 WebSocket 二进制帧推给前端 MSE.

独立于 InferenceSync(推理流 JSON 推送):本转发器只处理"带标注的视频流",
推理流(结构化面板数据:进度/流程/语音/评价)仍走 InferenceSync→send_text.

两条流(front/pop)各一个消费线程,从各自 Stream 末尾持续 xread,
按 init/media/end 段类型调 ws_handler.send_vis_chunk 转发.
"""
import logging
import threading
import time
from typing import Any, List

from core.vis_encoder import KEY_PREFIX

logger = logging.getLogger("web.vis_forwarder")


class VisStreamForwarder:
    """消费 inference:vis_stream:{front|pop} → ws_handler.send_vis_chunk."""

    def __init__(
        self,
        ws_handler: Any,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
    ) -> None:
        """初始化视觉流转发器.

        Args:
            ws_handler: WebSocket 处理器，提供 send_vis_chunk 推送接口.
            redis_host: Redis 主机地址.
            redis_port: Redis 端口.
            redis_db: Redis 数据库编号.
        """
        import redis

        self._ws = ws_handler
        # 二进制 payload,decode_responses=False
        self._redis = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=False
        )
        self._views = ("front", "pop")
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []

    def start(self) -> None:
        """启动 front/pop 两条消费线程."""
        self._stop.clear()  # 重启时清停止信号,否则 _consume 线程 while 条件立即为假秒退
        for view in self._views:
            thread = threading.Thread(
                target=self._consume, args=(view,), daemon=True, name=f"vis_fwd_{view}"
            )
            thread.start()
            self._threads.append(thread)
        logger.info("VisStreamForwarder 启动,消费 front/pop")

    def _consume(self, view: str) -> None:
        """持续消费单个视角的 fMP4 Redis Stream 并转发到 WebSocket.

        Args:
            view: 视角名（front/pop）.
        """
        key = KEY_PREFIX + view
        last_id = "0-0"
        while not self._stop.is_set():
            try:
                stream_entries = self._redis.xread({key: last_id}, count=64, block=500)
                if not stream_entries:
                    continue
                for _stream_key, entries in stream_entries:
                    for entry_id, fields in entries:
                        last_id = (
                            entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                        )
                        seg_type = fields.get(b"type", b"").decode()
                        data = fields.get(b"data", b"")
                        if seg_type == "end":
                            # 转发 end 后继续消费(不 return 退出):
                            # 实测出现过 end 提前落流的间歇现象(触发源未定位), 线程退出会导致
                            # 其后全部 media 段无人转发→前端该视角断供失步; 继续读则即便 end
                            # 提前出现, 后续 media 仍能送达前端(前端 EOS 有宽限期兜底)
                            self._ws.send_vis_chunk(view, "end", b"")
                            continue
                        self._ws.send_vis_chunk(view, seg_type, data)
            except Exception as error:
                logger.error(f"VisStreamForwarder[{view}] 消费异常: {error}")
                time.sleep(0.5)

    def stop(self) -> None:
        """停止全部消费线程."""
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5)
        logger.info("VisStreamForwarder 已停止")
