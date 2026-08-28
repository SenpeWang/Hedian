"""
HTTP 服务器模块 (FastAPI).

负责 FastAPI 路由和 Web 服务。
"""
import asyncio
import json
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
    pipeline_state = {"status": "idle"}

    def _r():
        """新建 Redis 连接（连接参数从 config 注入）."""
        from core.redis_conn import get_redis_client
        return get_redis_client(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            db=config.get("redis_db", 0),
        )

    # 保存到 app.state 供外部访问
    app.state.ws_handler = ws_handler
    app.state.pipeline_state = pipeline_state
    app.state.inference_sync = inference_sync
    # 注入总时长/状态/同步器, 供 WSHandler connect 补发状态快照(前端刷新即恢复进度)
    ws_handler.set_state_refs(config.get("duration", 0.0), pipeline_state, inference_sync)

    @app.post("/start")
    async def start(request: Request):
        """启动流水线（幂等：已在运行时拒绝重复启动."""
        if pipeline_state["status"] == "running":
            logger.warning("拒绝重复启动：流水线已在运行")
            return {"status": "already_running"}

        # 一次性清理所有推理流/事件/进度 key（scan_iter 遍历时删除安全）
        r = _r()
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

        r = _r()
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

    @app.post("/reset")
    async def reset(request: Request):
        """页面刷新时调用:kill 推理子进程 + 清空 Redis 状态 + 清 init 缓存,回到干净 idle."""
        stop_fn = getattr(app.state, "stop_pipeline", None)
        stopped_count = stop_fn() if stop_fn else 0
        # 彻底清空所有 Redis 残留(与 /start 一致), 不遗漏非前缀 key
        r = _r()
        r.flushdb()
        if inference_sync:
            inference_sync.reset()
        ws_handler.clear_vis_cache()
        pipeline_state["status"] = "idle"
        logger.info(f"页面刷新重置:已终止 {stopped_count} 个推理子进程,清空状态,切回 idle")
        return {"status": "reset", "terminated_processes": stopped_count}

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

    async def _video_response(prefix: str, name: str):
        """流式提供视角视频文件（支持 HTTP Range 206 硬解与原生声音）."""
        base_dir = Path(__file__).resolve().parent.parent
        for ext in ("mp4", "mpg"):
            candidate = base_dir / "data/videos" / f"{name}.{ext}"
            if candidate.is_file():
                return FileResponse(str(candidate), media_type="video/mp4")
        return JSONResponse({"error": f"{prefix} video not found"},
                            status_code=404)

    @app.get("/api/video/front")
    async def get_video_front():
        """流式提供前置视角视频流."""
        return await _video_response("front", "camFRONT")

    @app.get("/api/video/pop")
    async def get_video_pop():
        """流式提供俯视视角视频流."""
        return await _video_response("pop", "camPOP")

    @app.get("/status")
    async def status():
        """状态."""
        return {
            "pipeline": pipeline_state["status"],
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
