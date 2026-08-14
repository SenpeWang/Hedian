"""
HTTP 服务器模块 (FastAPI).

负责 FastAPI 路由和 Web 服务。
"""
import asyncio
import logging
import time
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from web.ws_handler import WSHandler

logger = logging.getLogger("web.server")


def create_app(
    config: dict,
    event_bus,
    paths,
    inference_sync=None,
) -> FastAPI:
    """创建 FastAPI 应用."""
    app = FastAPI(title="Hedian A_DemoSrc")

    # WebSocket 处理器
    ws_handler = WSHandler()

    # 流水线状态
    pipeline_state = {"status": "idle", "thread": None}

    # 保存到 app.state 供外部访问
    app.state.ws_handler = ws_handler
    app.state.pipeline_state = pipeline_state
    app.state.inference_sync = inference_sync

    @app.get("/")
    async def index():
        """首页 — 由 StaticFiles 兜底，此处仅为显式声明."""
        dist = Path(__file__).parent.parent / "frontend" / "dist"
        return FileResponse(str(dist / "index.html"))

    @app.post("/start")
    async def start(request: Request):
        """启动流水线（幂等：已在运行时拒绝重复启动."""
        if pipeline_state["status"] == "running":
            logger.warning("拒绝重复启动：流水线已在运行")
            return {"status": "already_running"}

        from core.redis_conn import get_redis_client
        r = get_redis_client(
            host=config.get("_redis_host", "localhost"),
            port=config.get("_redis_port", 6379),
            db=config.get("_redis_db", 0),
        )
        # 一次性清理所有推理流/事件/进度 key（scan_iter 遍历时删除安全）
        r.flushdb()
        ws_handler.reset()
        logger.info("已执行 r.flushdb() 彻底清空 Redis 历史残留数据")

        sig_time = time.time()
        r.set("pipeline:start_signal", str(sig_time), ex=3600)

        from datetime import datetime
        new_run_id = datetime.fromtimestamp(sig_time).strftime("%Y%m%d_%H%M%S")
        config["run_id"] = new_run_id

        if inference_sync:
            inference_sync.reset()

        event_bus.publish("pipeline.start", {"run_id": new_run_id},
                          ts=sig_time)

        pipeline_state["status"] = "running"
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"启动信号已设置，run_id={new_run_id}，来源: {client_host}")
        return {"status": "started"}

    @app.post("/stop")
    async def stop(request: Request):
        """停止."""
        stop_fn = getattr(app.state, "stop_pipeline", None)
        stopped_count = 0
        if stop_fn:
            stopped_count = stop_fn()

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

        if inference_sync:
            inference_sync.reset()

        pipeline_state["status"] = "idle"
        ws_handler.push({
            "source": "done",
            "tag": "stop",
            "data": {
                "reason": "user_stopped"
            }
        })
        logger.info(f"收到主动停止请求，已终止 {stopped_count} 个运行中的推理子进程，显存与状态切回 idle")
        return {"status": "stopped", "terminated_processes": stopped_count}

    @app.websocket("/ws/data")
    async def websocket_data(websocket: WebSocket):
        """websocket数据."""
        ws_handler.set_event_loop(asyncio.get_running_loop())
        await ws_handler.connect(websocket)
        try:
            while True:
                # 保持双工接收心跳与前端播放进度上报
                client_text = await websocket.receive_text()
                try:
                    client_msg = json.loads(client_text)
                    if isinstance(client_msg, dict) and client_msg.get('type') == 'playback_progress':
                        current_sec = client_msg.get('current_sec', 0.0)
                        ws_handler.update_playback_sec(current_sec)
                except Exception:
                    pass
        except WebSocketDisconnect:
            ws_handler.disconnect(websocket)
        except Exception as e:
            logger.debug(f"WebSocket 非预期断开: {e}")
            ws_handler.disconnect(websocket)

    @app.get("/api/audio/stream")
    async def get_audio_stream():
        """提供从 camFRONT 提取的完整音频文件，确保 100% 毫秒级 200 OK 秒开播放."""
        import os
        base_dir = Path(__file__).resolve().parent.parent
        candidates = [
            base_dir / "data/videos/camFRONT_audio.wav",
            Path("/home/wangshengping/Hedian/A_DemoSrc/data/videos/camFRONT_audio.wav"),
        ]
        for c in candidates:
            if c.is_file():
                logger.info(f"[AudioStream] 成功找到音频文件: {c}")
                return FileResponse(str(c), media_type="audio/wav")
        
        logger.error(f"[AudioStream] 未找到任何候选音频文件, checked: {candidates}")
        return JSONResponse({"error": "audio file not found"}, status_code=404)

    @app.get("/status")
    async def status():
        """状态."""
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
            "ws_clients": ws_handler.get_client_count(),
        }

    @app.get("/api/config")
    async def get_config():
        """获取配置."""
        return config

    @app.get("/api/modules")
    async def get_modules():
        """获取modules."""
        return {"modules": config.get("modules", {})}

    class NoCacheStaticFiles(StaticFiles):
        """禁用浏览器缓存的静态文件服务."""

        async def get_response(self, path: str, scope):
            """获取response."""
            response = await super().get_response(path, scope)
            response.headers[
                "Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

    # 静态文件兜底（所有未匹配路由 → dist/）
    dist_dir = Path(__file__).parent.parent / "frontend" / "dist"
    if dist_dir.is_dir():
        app.mount("/",
                  NoCacheStaticFiles(directory=str(dist_dir), html=True),
                  name="static")

    return app
