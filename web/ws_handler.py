"""WebSocket 客户端连接与实时数据推送管理器.

负责维护前端双工 WebSocket 连接、打包二进制视频帧与 JSON 元数据，
并支持独立文本直推与前端播放时钟感知。
"""
import asyncio
import json
import logging
import struct
import threading
from typing import Any, Dict, List, Optional
from fastapi import WebSocket

logger = logging.getLogger("web.ws_handler")

# 视角标识常量（1 字节 uint8）
VIEW_ID_FRONT = 0  # camFRONT 前置广角视角
VIEW_ID_POP = 1    # camPOP 行为特写视角

_VIEW_KEY_TO_ID: Dict[str, int] = {
    "video_front": VIEW_ID_FRONT,
    "video_pop": VIEW_ID_POP,
}

# 统一二进制消息协议版本
_BINARY_VERSION: int = 1


class WSHandler:
    """WebSocket 客户端推送与状态连接管理器.

    单条二进制消息同时承载视频帧与 JSON 文字元数据，保证两者原子到达同一 batch。
    同时支持双工接收前端播放时间戳（playback_progress），为后端流式评估提供时钟触发依据。
    """

    def __init__(self):
        """初始化 WebSocket 处理器."""
        self._active_connections: List[WebSocket] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._current_playback_sec: float = 0.0
        self._playback_lock = threading.Lock()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """设置异步事件循环.

        Args:
            loop (asyncio.AbstractEventLoop): FastAPI 异步主事件循环.
        """
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        """接受并注册新的客户端 WebSocket 连接.

        Args:
            websocket (WebSocket): 客户端连接对象.
        """
        await websocket.accept()
        self._active_connections.append(websocket)
        logger.info(f"WebSocket 客户端已连接，当前在线: {len(self._active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """注销已断开的客户端 WebSocket 连接.

        Args:
            websocket (WebSocket): 待注销的连接对象.
        """
        if websocket in self._active_connections:
            self._active_connections.remove(websocket)
            logger.info(f"WebSocket 客户端已断开，当前在线: {len(self._active_connections)}")

    def reset(self) -> None:
        """重置播放时钟状态."""
        with self._playback_lock:
            self._current_playback_sec = 0.0
        logger.info("WSHandler 播放时钟已重置为 0.0s")

    def update_playback_sec(self, current_sec: float) -> None:
        """更新前端实际上报的画面播放时间戳（秒）.

        Args:
            current_sec (float): 当前视频画面渲染所在的全局秒数.
        """
        with self._playback_lock:
            self._current_playback_sec = max(self._current_playback_sec, float(current_sec))

    def get_playback_sec(self) -> float:
        """获取当前前端实际上报的画面播放时间戳（秒）.

        Returns:
            float: 前端当前正在播放的全局秒数.
        """
        with self._playback_lock:
            return self._current_playback_sec

    def push(self, event: Optional[Dict[str, Any]]) -> None:
        """推送视频与结构化元数据 batch 到所有在线客户端.

        Args:
            event (Optional[Dict[str, Any]]): 包含 globalSec、视频帧及各源数据的批次字典；
                若为 None 表示本轮推理结束，发送 done 哨兵文本。
        """
        if not self._active_connections or not self._loop:
            return

        # done 哨兵：走文本帧通知前端 EOF
        if not event:
            done_message = json.dumps({"source": "done"})
            for connection in list(self._active_connections):
                try:
                    asyncio.run_coroutine_threadsafe(
                        connection.send_text(done_message), self._loop
                    )
                except Exception as send_error:
                    logger.error(f"推送 WebSocket done 消息失败: {send_error}")
            return

        batch = event
        global_sec = float(batch.get("globalSec", 0.0))

        # 提取视频帧
        frames: List[Dict[str, Any]] = []
        for view_key, view_id in _VIEW_KEY_TO_ID.items():
            if view_key in batch:
                val = batch.pop(view_key)
                frame_dict = val[0] if (isinstance(val, list) and len(val) > 0) else (val if isinstance(val, dict) else None)
                if frame_dict and isinstance(frame_dict, dict):
                    frame_data = frame_dict.get("data", {}).get("frame_data")
                    if frame_data:
                        try:
                            jpeg_bytes = frame_data.encode("latin1") if isinstance(frame_data, str) else frame_data
                            frames.append({"view_id": view_id, "jpeg": jpeg_bytes})
                        except Exception as frame_error:
                            logger.error(f"提取视频帧失败 [{view_key}]: {frame_error}")

        # 剩余轻量元数据 JSON
        meta_json_bytes = json.dumps(batch, ensure_ascii=False).encode("utf-8")

        # 打包单条二进制协议包
        try:
            buffer = bytearray()
            buffer += struct.pack("!BfB", _BINARY_VERSION, global_sec, len(frames))
            for frame_item in frames:
                jpeg_data = frame_item["jpeg"]
                buffer += struct.pack("!BI", frame_item["view_id"], len(jpeg_data))
                buffer += jpeg_data
            buffer += struct.pack("!I", len(meta_json_bytes))
            buffer += meta_json_bytes
            binary_packet = bytes(buffer)

            for connection in list(self._active_connections):
                try:
                    asyncio.run_coroutine_threadsafe(
                        connection.send_bytes(binary_packet), self._loop
                    )
                except Exception as push_error:
                    logger.error(f"推送 WebSocket 二进制消息失败: {push_error}")
        except Exception as pack_error:
            logger.error(f"打包二进制消息失败: {pack_error}")

    def push_text(self, event: Dict[str, Any]) -> None:
        """推送纯文本 JSON 事件到所有客户端（用于评估报告流式直推等高实时事件）.

        与二进制视频 batch 物理解耦，不携带 globalSec，不抢占视频打包通道。

        Args:
            event (Dict[str, Any]): 待推送的 JSON 事件字典.
        """
        if not self._active_connections or not self._loop:
            return
        try:
            message_text = json.dumps(event, ensure_ascii=False)
            for connection in list(self._active_connections):
                try:
                    asyncio.run_coroutine_threadsafe(
                        connection.send_text(message_text), self._loop
                    )
                except Exception as send_error:
                    logger.error(f"推送 WebSocket 文本消息失败: {send_error}")
        except Exception as json_error:
            logger.error(f"序列化文本事件失败: {json_error}")

    def get_client_count(self) -> int:
        """获取当前在线 WebSocket 客户端数量.

        Returns:
            int: 在线客户端数.
        """
        return len(self._active_connections)
