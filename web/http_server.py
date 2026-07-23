"""
HTTP 服务器模块

负责 Flask 路由和 Web 服务。
"""
import os
import json
import logging
from pathlib import Path
from typing import Callable

from flask import Flask, Response, request, jsonify, send_file

from web.sse_handler import SSEHandler

logger = logging.getLogger("web.server")


def create_app(
    config: dict,
    event_bus,
    registry,
    paths,
    pipeline_runner: Callable = None,
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

    @app.route("/")
    def index():
        """首页"""
        return app.send_static_file("index.html")

    @app.route("/start", methods=["POST"])
    def start():
        """启动流水线：清理 Redis 缓存后设置带时间戳的代际信号，动态更新 run_id"""
        import time as _time
        from core.redis_conn import get_redis_client
        r = get_redis_client(
            host=config.get("_redis_host", "localhost"),
            port=config.get("_redis_port", 6379),
            db=config.get("_redis_db", 0),
        )
        # 清除上一次的缓存数据
        for key in r.scan_iter("inference:*"):
            r.delete(key)
        for key in r.scan_iter("module:*"):
            r.delete(key)
        for key in r.scan_iter("pipeline:*"):
            r.delete(key)
        for key in r.scan_iter("gaze:*"):
            r.delete(key)
        
        sig_time = _time.time()
        # 设置代际信号
        r.set("pipeline:start_signal", str(sig_time), ex=3600)
        
        # 动态生成本轮推理的 run_id，彻底隔绝物理目录干扰
        from datetime import datetime
        new_run_id = datetime.fromtimestamp(sig_time).strftime("%Y%m%d_%H%M%S")
        config["run_id"] = new_run_id
        
        # 广播新推理代际启动事件，驱动规则记录与评价器就地重置路径
        event_bus.publish("pipeline.start", {"run_id": new_run_id}, ts=sig_time)
        
        pipeline_state["status"] = "running"
        logger.info(f"启动信号已设置，生成动态运行批次 run_id={new_run_id}，来源: {request.remote_addr}")
        return jsonify({"status": "started"})

    @app.route("/data")
    def data_stream():
        """推理流 SSE"""
        logger.info(f"SSE /data 连接建立，来源: {request.remote_addr}")

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

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.route("/audio")
    def audio():
        """音频文件：严格只动态读取当前运行批次下的音频"""
        run_id = config.get("run_id")
        if run_id:
            wav = str(paths.get_result_path(run_id=run_id, module="voice", filename="audio.wav"))
            if os.path.exists(wav):
                return send_file(wav, mimetype="audio/wav")
        return "", 404

    @app.route("/status")
    def status():
        """获取状态"""
        from core.redis_conn import get_redis_client
        r = get_redis_client(
            host=config.get("_redis_host", "localhost"),
            port=config.get("_redis_port", 6379),
            db=config.get("_redis_db", 0),
        )
        redis_status = r.get("pipeline:status")

        status_val = pipeline_state["status"]
        # Redis 显示 done 则重置为空闲
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
    app.pipeline_state = pipeline_state

    return app

