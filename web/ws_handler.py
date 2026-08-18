"""WebSocket 客户端连接与实时数据推送管理器.

负责维护前端双工 WebSocket 连接,推送两类数据:
  1. 结构化推理流(纯 JSON 文本帧):进度/流程/语音/评价等面板数据
  2. 视频流 fMP4 段(二进制帧):带标注的 fragmented MP4,前端 MSE 播放
并支持独立文本直推与前端播放时钟感知.
"""
import asyncio
import json
import logging
import threading
from typing import Any, Dict, List, Optional
from fastapi import WebSocket

logger = logging.getLogger("web.ws_handler")

# 视频流二进制帧头:1 字节 channel(front=0/pop=1) + 1 字节 type(init=0/media=1/end=2)
_CHANNEL_FRONT = 0
_CHANNEL_POP = 1
_TYPE_CODE = {"init": 0, "media": 1, "end": 2}


class WSHandler:
    """WebSocket 客户端推送与状态连接管理器."""

    def __init__(self):
        """初始化 WebSocket 处理器."""
        self._active_connections: List[WebSocket] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._current_playback_sec: float = 0.0
        self._playback_lock = threading.Lock()
        # 视频流 init 段缓存:view -> fMP4 init bytes;新连接接入时立即下发,
        # 保证后到的浏览器也能从 init 开始正确解码 MSE。
        self._vis_init_cache: Dict[str, bytes] = {}
        # 状态快照注入: 总时长/流水线状态/同步器, connect 时补发给前端(刷新即恢复进度)
        self._total_duration: float = 0.0
        self._pipeline_state: Optional[Dict[str, Any]] = None
        self._inference_sync = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """设置异步事件循环.

        Args:
            loop (asyncio.AbstractEventLoop): FastAPI 异步主事件循环.
        """
        self._loop = loop

    def set_state_refs(self, total_duration: float, pipeline_state, inference_sync=None) -> None:
        """注入总时长/流水线状态/同步器, 供 connect 补发状态快照."""
        self._total_duration = float(total_duration or 0.0)
        self._pipeline_state = pipeline_state
        self._inference_sync = inference_sync

    async def send_status_snapshot(self, websocket: WebSocket) -> None:
        """连接时补发当前流水线状态快照(总时长+当前推理秒+状态).

        前端刷新页面后 JS 状态清零, 此快照让前端立即拿到稳定的 totalDuration
        (不依赖前端 video 是否加载) 与当前 globalSec, 推理进度条可正确恢复.
        """
        total = self._total_duration or 0.0
        status_s = (self._pipeline_state or {}).get("status", "idle")
        if status_s == "running" and self._inference_sync is not None:
            try:
                g = self._inference_sync._compute_global_sec()
                # 排除 inf/nan/负
                if g == float("inf") or not (g == g) or g < 0:
                    g = 0.0
            except Exception:
                g = 0.0
        else:
            # idle: 未开始或已重置, 进度 0(刷新=重新开始)
            g = 0.0
        msg = {"source": "status", "totalDuration": total, "globalSec": g, "status": status_s}
        try:
            await websocket.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"补发状态快照失败: {e}")

    async def connect(self, websocket: WebSocket) -> None:
        """接受并注册新的客户端 WebSocket 连接,并补发已缓存的视频流 init 段."""
        await websocket.accept()
        self._active_connections.append(websocket)
        logger.info(f"WebSocket 客户端已连接，当前在线: {len(self._active_connections)}")
        # 新连接:立即下发已缓存的视频流 init 段(若本轮推理已产出)
        await self._send_cached_vis_inits(websocket)
        # 补发状态快照: 总时长+当前推理秒+状态, 前端刷新即恢复进度条
        await self.send_status_snapshot(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """注销已断开的客户端 WebSocket 连接.

        Args:
            websocket (WebSocket): 待注销的连接对象.
        """
        if websocket in self._active_connections:
            self._active_connections.remove(websocket)
            logger.info(f"WebSocket 客户端已断开，当前在线: {len(self._active_connections)}")

    def reset(self) -> None:
        """重置播放时钟状态与视频流 init 缓存."""
        with self._playback_lock:
            self._current_playback_sec = 0.0
        self._vis_init_cache.clear()
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
        """推送结构化元数据 batch 到所有在线客户端 (纯 JSON 文本帧).

        Args:
            event (Optional[Dict[str, Any]]): 包含 globalSec 及各源数据的批次字典；
                若为 None 表示本轮推理结束，发送 done 哨兵文本。
        """
        if not self._active_connections or not self._loop:
            return

        # done 哨兵：通知前端推理完成
        if not event:
            done_message = json.dumps({"source": "done"}, ensure_ascii=False)
            for connection in list(self._active_connections):
                try:
                    asyncio.run_coroutine_threadsafe(
                        connection.send_text(done_message), self._loop
                    )
                except Exception as send_error:
                    logger.error(f"推送 WebSocket done 消息失败: {send_error}")
            return

        try:
            message_text = json.dumps(event, ensure_ascii=False)
            for connection in list(self._active_connections):
                try:
                    asyncio.run_coroutine_threadsafe(
                        connection.send_text(message_text), self._loop
                    )
                except Exception as send_error:
                    logger.error(f"推送 WebSocket 消息失败: {send_error}")
        except Exception as json_error:
            logger.error(f"序列化 WebSocket 事件失败: {json_error}")

    def push_text(self, event: Dict[str, Any]) -> None:
        """推送纯文本 JSON 事件到所有客户端（用于评估报告流式直推等高实时事件）.

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

    # ── 视频流(fMP4)二进制推送 ───────────────────────────────────

    @staticmethod
    def _build_vis_payload(view: str, seg_type: str, data: bytes) -> bytes:
        """构造视频流二进制帧:[channel][type_code] + fMP4 段."""
        channel = _CHANNEL_FRONT if view == "front" else _CHANNEL_POP
        type_code = _TYPE_CODE.get(seg_type, 1)
        return bytes([channel, type_code]) + data

    async def _send_cached_vis_inits(self, websocket: WebSocket) -> None:
        """向新连接补发已缓存的 init 段(保证后到连接能从 init 解码)."""
        for view, data in self._vis_init_cache.items():
            try:
                await websocket.send_bytes(self._build_vis_payload(view, "init", data))
            except Exception as e:
                logger.error(f"补发 init 段失败 [{view}]: {e}")

    def send_vis_chunk(self, view: str, seg_type: str, data: bytes) -> None:
        """把一段 fMP4(init/media/end)以二进制帧推给所有在线客户端.

        由 VisStreamForwarder 后台线程调用;init 段同时缓存以备新连接补发.
        """
        # init 段先缓存(无论当前有无连接),保证后到连接能补发
        if seg_type == "init":
            self._vis_init_cache[view] = data
        if not self._active_connections or not self._loop:
            return
        payload = self._build_vis_payload(view, seg_type, data)
        for connection in list(self._active_connections):
            try:
                asyncio.run_coroutine_threadsafe(
                    connection.send_bytes(payload), self._loop
                )
            except Exception as send_error:
                logger.error(f"推送 vis chunk 失败 [{view}/{seg_type}]: {send_error}")

    def clear_vis_cache(self) -> None:
        """清空视频流 init 缓存(新一轮推理开始前调用)."""
        self._vis_init_cache.clear()

    def get_client_count(self) -> int:
        """获取当前在线 WebSocket 客户端数量.

        Returns:
            int: 在线客户端数.
        """
        return len(self._active_connections)
