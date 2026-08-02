import asyncio
import json
import logging
import struct
from typing import List, Dict, Any, Optional
from fastapi import WebSocket

logger = logging.getLogger("web.ws")

# 视角 ID 映射表 (0=front, 1=bup, 2=pop)
VIEW_MAP = {
    "video_front": 0,
    "video_bup": 1,
    "video_pop": 2,
}

# 统一二进制消息协议版本
_BINARY_VERSION = 1


class WSHandler:
    """WebSocket 客户端推送连接管理器

    单条二进制消息同时承载视频帧与 JSON 文字元数据，保证两者原子到达同一 batch，
    消除"二进制帧与文本帧到达顺序无保证"的竞态。

    消息布局（大端）：
        [1B version]
        [4B globalSec (float32)]
        [1B view_count N (0..3)]
        重复 N 次:
            [1B view_id (0=front/1=bup/2=pop)]
            [4B frame_len (uint32)]
            [frame_len bytes JPEG]
        [4B json_len (uint32)]
        [json_len bytes UTF-8 JSON 元数据]
    """

    def __init__(self):
        self._active_connections: List[WebSocket] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self._active_connections.append(websocket)
        logger.info(f"WebSocket 客户端已连接，当前在线: {len(self._active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self._active_connections:
            self._active_connections.remove(websocket)
            logger.info(f"WebSocket 客户端已断开，当前在线: {len(self._active_connections)}")

    def push(self, event: Optional[Dict[str, Any]]):
        """推送 batch 到所有客户端

        Args:
            event: batch dict（含 globalSec + 各 source 事件 + video_* 帧数据）；
                   None 表示 done 哨兵，发送 {"source":"done"} 文本。
        """
        if not self._active_connections or not self._loop:
            return

        # done 哨兵：无负载，走文本帧（语义清晰，前端仅凭文本分支识别 EOF）
        if not event:
            done_msg = json.dumps({"source": "done"})
            for connection in list(self._active_connections):
                try:
                    asyncio.run_coroutine_threadsafe(connection.send_text(done_msg), self._loop)
                except Exception as e:
                    logger.error(f"推送 WebSocket done 消息失败: {e}")
            return

        # 浅拷贝 batch，抽取视频帧进行二进制打包，文字数据放尾部 JSON
        batch = dict(event)
        global_sec = float(batch.get("globalSec", 0.0))

        # 收集三路视频帧（每路至多 1 帧，已由 InferenceSync 去重）
        frames: List[Dict[str, Any]] = []  # [{view_id, jpeg_bytes}, ...]
        for view_key, view_id in VIEW_MAP.items():
            frames_list = batch.pop(view_key, None)
            if frames_list and isinstance(frames_list, list) and len(frames_list) > 0:
                frame_data = frames_list[0].get("data", {}).get("frame_data")
                if frame_data:
                    try:
                        # 原生 latin1 零 CPU 转换还原原始 JPEG 二进制 bytes
                        jpeg_bytes = frame_data.encode('latin1') if isinstance(frame_data, str) else frame_data
                        frames.append({"view_id": view_id, "jpeg": jpeg_bytes})
                    except Exception as ex:
                        logger.error(f"提取视频帧失败 [{view_key}]: {ex}")

        # 剩余轻量元数据 JSON（含 voice/gaze/tracking/progress/flow 等）
        meta_json = json.dumps(batch, ensure_ascii=False).encode('utf-8')

        # 打包成单条二进制消息
        try:
            # Header: [1B version][4B globalSec float32][1B view_count]
            buf = bytearray()
            buf += struct.pack("!BfB", _BINARY_VERSION, global_sec, len(frames))
            # 各视角帧: [1B view_id][4B frame_len][JPEG bytes]
            for f in frames:
                jpeg = f["jpeg"]
                buf += struct.pack("!BI", f["view_id"], len(jpeg))
                buf += jpeg
            # 尾部 JSON 元数据: [4B json_len][JSON bytes]
            buf += struct.pack("!I", len(meta_json))
            buf += meta_json
            binary_pkt = bytes(buf)

            for connection in list(self._active_connections):
                try:
                    asyncio.run_coroutine_threadsafe(
                        connection.send_bytes(binary_pkt), self._loop
                    )
                except Exception as e:
                    logger.error(f"推送 WebSocket 二进制消息失败: {e}")
        except Exception as ex:
            logger.error(f"打包二进制消息失败: {ex}")

    def push_text(self, event: Dict[str, Any]) -> None:
        """推送纯文本 JSON 事件到所有客户端（用于评估报告等高实时性直推事件）

        与 push() 的二进制 batch 协议物理隔离：直推事件无视频帧、无 globalSec，
        不应被塞进二进制打包流程（否则前端 globalSec 被拉回 0、meta 结构不匹配）。
        前端 onmessage 文本分支按 tag 识别（segment_report / segment_report_stream）。
        """
        if not self._active_connections or not self._loop:
            return
        try:
            msg = json.dumps(event, ensure_ascii=False)
            for connection in list(self._active_connections):
                try:
                    asyncio.run_coroutine_threadsafe(connection.send_text(msg), self._loop)
                except Exception as e:
                    logger.error(f"推送 WebSocket 文本消息失败: {e}")
        except Exception as ex:
            logger.error(f"序列化文本事件失败: {ex}")

    def get_client_count(self) -> int:
        return len(self._active_connections)
