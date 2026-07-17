"""
HTTP 服务器模块

负责 Flask 路由和 Web 服务。
"""
import os
import json
import logging
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

from flask import Flask, Response, request, jsonify, send_file

from web.sse_handler import SSEHandler

logger = logging.getLogger("web.server")


def create_app(
    config: dict,
    event_bus,
    registry,
    paths,
    pipeline_runner: Callable = None,
    frame_queues: dict = None,
) -> Flask:
    """
    创建 Flask 应用

    Args:
        config: 配置字典
        event_bus: 消息总线
        registry: 制度注册表
        paths: 路径配置
        pipeline_runner: 流水线运行函数

    Returns:
        Flask 应用
    """
    app = Flask(
        __name__,
        static_folder=str(Path(__file__).parent / "static"),
    )

    # SSE 处理器
    sse_handler = SSEHandler()

    # 流水线状态
    pipeline_state = {"status": "idle", "thread": None}

    # 视频帧队列（从外部传入，与模块共享）
    if frame_queues is None:
        frame_queues = {
            "tracker": queue.Queue(maxsize=8),
            "behavior": queue.Queue(maxsize=8),
        }

    @app.route("/")
    def index():
        """首页"""
        return app.send_static_file("index.html")

    @app.route("/start", methods=["POST"])
    def start():
        """启动流水线（设置启动信号）— 仅接受 POST"""
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        r.delete("pipeline:status")  # 启动新推理时，清除上一次的完成标志
        r.set("pipeline:start_signal", "start", ex=3600)
        r.close()
        pipeline_state["status"] = "running"
        logger.info(f"启动信号已设置，来源: {request.remote_addr}")
        return jsonify({"status": "started"})

    @app.route("/events")
    def events():
        """SSE 事件流"""
        def generate():
            client_queue = sse_handler.add_client()
            try:
                while True:
                    try:
                        event = client_queue.get(timeout=25)
                    except Exception:
                        yield ": keepalive\n\n"
                        continue

                    if event is None:
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        break

                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            finally:
                sse_handler.remove_client(client_queue)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.route("/stream")
    def stream():
        """MOT 视频流"""
        def generate():
            q = frame_queues["tracker"]
            while True:
                try:
                    data = q.get(timeout=5)
                except Exception:
                    continue
                if data is None:
                    break
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/stream2")
    def stream2():
        """行为检测视频流"""
        def generate():
            q = frame_queues["behavior"]
            while True:
                try:
                    data = q.get(timeout=5)
                except Exception:
                    continue
                if data is None:
                    break
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/audio")
    def audio():
        """音频文件"""
        wav = str(paths.data_root / "raw_from_video.wav")
        if os.path.exists(wav):
            return send_file(wav, mimetype="audio/wav")
        return "", 404

    @app.route("/status")
    def status():
        """获取状态"""
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        redis_status = r.get("pipeline:status")
        r.close()

        status_val = pipeline_state["status"]
        # 如果 Redis 中 pipeline:status 是 done，代表推理已经全部跑完，重置为空闲
        if redis_status == "done":
            status_val = "idle"
            pipeline_state["status"] = "idle"

        return jsonify({
            "pipeline": status_val,
            "sse_clients": sse_handler.get_client_count(),
        })

    @app.route("/api/config")
    def get_config():
        """获取配置"""
        return jsonify(config)

    @app.route("/api/modules")
    def get_modules():
        """获取模块列表"""
        return jsonify({
            "modules": config.get("modules", {}),
        })

    # 保存引用供外部使用
    app.sse_handler = sse_handler
    app.frame_queues = frame_queues
    app.pipeline_state = pipeline_state

    return app


def _run_pipeline(
    pipeline_runner: Callable,
    sse_handler: SSEHandler,
    pipeline_state: dict,
) -> None:
    """
    运行流水线

    Args:
        pipeline_runner: 流水线运行函数
        sse_handler: SSE 处理器
        pipeline_state: 流水线状态
    """
    try:
        pipeline_runner()
    except Exception as e:
        logger.error(f"流水线错误: {e}", exc_info=True)
        sse_handler.push({"type": "error", "message": str(e)})
    finally:
        pipeline_state["status"] = "idle"
