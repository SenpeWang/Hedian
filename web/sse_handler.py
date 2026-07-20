"""
SSE 推送处理模块

负责管理 SSE 客户端连接和事件推送。
"""
import queue
import threading
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("web.sse")


class SSEHandler:
    """
    SSE 推送处理器

    管理 SSE 客户端连接，推送事件到所有连接的客户端。
    """

    def __init__(self):
        """初始化 SSE 处理器"""
        self._clients = []
        self._lock = threading.Lock()

    def add_client(self) -> queue.Queue:
        """
        添加 SSE 客户端

        Returns:
            客户端队列
        """
        client_queue = queue.Queue(maxsize=4096)
        with self._lock:
            self._clients.append(client_queue)
        logger.info(f"添加 SSE 客户端，当前 {len(self._clients)} 个")
        return client_queue

    def remove_client(self, client_queue: queue.Queue) -> None:
        """
        移除 SSE 客户端

        Args:
            client_queue: 客户端队列
        """
        with self._lock:
            if client_queue in self._clients:
                self._clients.remove(client_queue)
        logger.info(f"移除 SSE 客户端，当前 {len(self._clients)} 个")

    def push(self, event: Optional[Dict[str, Any]]) -> None:
        """推送事件到所有客户端"""
        with self._lock:
            if event and event.get("source") not in ("clock_sync", "progress", "video"):
                logger.debug(f"SSE推送: source={event.get('source')}, 客户端数={len(self._clients)}")
            for client_queue in self._clients:
                try:
                    client_queue.put_nowait(event)
                except queue.Full:
                    logger.warning("SSE 客户端队列已满")

    def get_client_count(self) -> int:
        """
        获取客户端数量

        Returns:
            客户端数量
        """
        with self._lock:
            return len(self._clients)
