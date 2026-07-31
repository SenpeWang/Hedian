"""
HTTP 服务器模块 (FastAPI)

负责 FastAPI 路由和 Web 服务。
"""
import os
import json
import logging
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from web.sse_handler import SSEHandler

logger = logging.getLogger("web.server")


def create_app(
    config: dict,
    event_bus,
    registry,
    paths,
    pipeline_runner: Callable = None,
    display_buffer=None,
) -> FastAPI:
    """创建 FastAPI 应用"""

    app = FastAPI(title="Hedian A_DemoSrc")

    # SSE 处理器
    sse_handler = SSEHandler()

    # 流水线状态
    pipeline_state = {"status": "idle", "thread": None}

    # 保存到 app.state 供外部访问
    app.state.sse_handler = sse_handler
    app.state.pipeline_state = pipeline_state
    app.state.display_buffer = display_buffer

    @app.get("/")
    async def index():
        """首页 — 由 StaticFiles 兜底，此处仅为显式声明"""
        dist = Path(__file__).parent.parent / "frontend" / "dist"
        return FileResponse(str(dist / "index.html"))

    @app.post("/start")
    async def start(request: Request):
        """启动流水线"""
        import time as _time
        from core.redis_conn import get_redis_client
        r = get_redis_client(
            host=config.get("_redis_host", "localhost"),
            port=config.get("_redis_port", 6379),
            db=config.get("_redis_db", 0),
        )
        for key in r.scan_iter("inference:*"):
            r.delete(key)
        for key in r.scan_iter("module:*"):
            r.delete(key)
        for key in r.scan_iter("pipeline:*"):
            r.delete(key)
        for key in r.scan_iter("gaze:*"):
            r.delete(key)

        sig_time = _time.time()
        r.set("pipeline:start_signal", str(sig_time), ex=3600)

        from datetime import datetime
        new_run_id = datetime.fromtimestamp(sig_time).strftime("%Y%m%d_%H%M%S")
        config["run_id"] = new_run_id

        if display_buffer:
            display_buffer.reset()

        event_bus.publish("pipeline.start", {"run_id": new_run_id}, ts=sig_time)

        pipeline_state["status"] = "running"
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"启动信号已设置，run_id={new_run_id}，来源: {client_host}")
        return {"status": "started"}

    @app.get("/data")
    async def data_stream(request: Request):
        """推理流 SSE"""
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"SSE /data 连接建立，来源: {client_host}")

        def generate():
            client_queue = sse_handler.add_client()
            try:
                while True:
                    try:
                        item = client_queue.get(timeout=25)
                    except Exception:
                        yield ": keepalive\n\n"
                        continue

                    if item is None:
                        yield f"data: {json.dumps({'source': 'done'})}\n\n"
                        break

                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            finally:
                sse_handler.remove_client(client_queue)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )



    @app.get("/api/audio/stream")
    async def get_audio_stream():
        """提供从 camFRONT 提取的完整音频文件，确保 100% 毫秒级 200 OK 秒开播放"""
        base_dir = Path(__file__).resolve().parent.parent
        cached_wav = base_dir / "data/videos/camFRONT_audio.wav"
        if cached_wav.is_file():
            return FileResponse(str(cached_wav), media_type="audio/wav")

        active_id = config.get("run_id")
        if active_id:
            wav_path = paths.get_result_dir(active_id) / "voice" / "audio.wav"
            if wav_path.is_file():
                return FileResponse(str(wav_path), media_type="audio/wav")

        return JSONResponse({"error": "audio file not found"}, status_code=404)

    @app.get("/status")
    async def status():
        """获取状态"""
        from core.redis_conn import get_redis_client
        r = get_redis_client(
            host=config.get("_redis_host", "localhost"),
            port=config.get("_redis_port", 6379),
            db=config.get("_redis_db", 0),
        )
        redis_status = r.get("pipeline:status")

        status_val = pipeline_state["status"]
        if redis_status == "done":
            status_val = "idle"
            pipeline_state["status"] = "idle"

        return {
            "pipeline": status_val,
            "sse_clients": sse_handler.get_client_count(),
        }

    @app.get("/api/config")
    async def get_config():
        return config

    @app.get("/api/modules")
    async def get_modules():
        return {"modules": config.get("modules", {})}

    # 静态文件兜底（所有未匹配路由 → dist/）
    dist_dir = Path(__file__).parent.parent / "frontend" / "dist"
    if dist_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")

    return app
