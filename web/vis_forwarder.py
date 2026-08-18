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

logger = logging.getLogger("web.vis_forwarder")

KEY_PREFIX = "inference:vis_stream:"


class VisStreamForwarder:
    """消费 inference:vis_stream:{front|pop} → ws_handler.send_vis_chunk."""

    def __init__(
        self,
        ws_handler,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
    ):
        self._ws = ws_handler
        import redis
        # 二进制 payload,decode_responses=False
        self._redis = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=False
        )
        self._views = ("front", "pop")
        self._stop = threading.Event()
        self._threads = []

    def start(self) -> None:
        self._stop.clear()  # 重启时清停止信号,否则 _consume 线程 while 条件立即为假秒退
        for view in self._views:
            t = threading.Thread(
                target=self._consume, args=(view,), daemon=True, name=f"vis_fwd_{view}"
            )
            t.start()
            self._threads.append(t)
        logger.info("VisStreamForwarder 启动,消费 front/pop")

    def _consume(self, view: str) -> None:
        key = KEY_PREFIX + view
        last_id = "0-0"
        while not self._stop.is_set():
            try:
                resp = self._redis.xread({key: last_id}, count=64, block=500)
                if not resp:
                    continue
                for _key, entries in resp:
                    for entry_id, fields in entries:
                        last_id = (
                            entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                        )
                        seg_type = fields.get(b"type", b"").decode()
                        data = fields.get(b"data", b"")
                        if seg_type == "end":
                            self._ws.send_vis_chunk(view, "end", b"")
                            return
                        self._ws.send_vis_chunk(view, seg_type, data)
            except Exception as e:
                logger.error(f"VisStreamForwarder[{view}] 消费异常: {e}")
                time.sleep(0.5)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)
        logger.info("VisStreamForwarder 已停止")
