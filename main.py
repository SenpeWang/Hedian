"""
核电站监护制合规检测系统 — 入口文件（多进程版）

架构:
  main.py        → 协调器（启动各模块进程）
  core/          → 核心框架（消息总线、配置、聚合器、模块基类）
  modules/       → 业务模块（语音、MOT、注视、行为）
  regulations/   → 制度层（监护制、自唱票）
  evaluation/    → 评估层（规则评估、大模型评估）
  web/           → 前端层（HTTP 服务器、SSE）

多进程架构:
  每个模块运行在独立进程中，通过 Redis 通信。
"""
import sys
import os
import argparse
import logging
import multiprocessing
import time
import redis
from datetime import datetime

# 解析命令行参数
parser = argparse.ArgumentParser(description="核电站监护制合规检测系统")
parser.add_argument("--gpu", type=str, default="0", help="GPU 编号 (默认: 0)")
parser.add_argument("--config", type=str, default=None, help="配置文件路径")
args = parser.parse_args()

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# 导入核心模块
from core.config_manager import ConfigManager
from core.path_manager import PathConfig
from core.logger import setup_logger

# Redis 配置
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

logger = setup_logger("main")

# 启动信号 Redis key
START_SIGNAL_KEY = "pipeline:start_signal"


def wait_for_start_signal(redis_host="localhost", redis_port=6379, redis_db=0):
    """等待用户点击'开始测试'的信号"""
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
    logger.info("等待用户点击'开始测试'...")
    while True:
        signal = r.get(START_SIGNAL_KEY)
        if signal == "start":
            r.delete(START_SIGNAL_KEY)  # 清除信号
            logger.info("收到启动信号，开始推理")
            return
        time.sleep(0.5)


def set_start_signal(redis_host="localhost", redis_port=6379, redis_db=0):
    """设置启动信号（由 Web 进程调用）"""
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
    r.set(START_SIGNAL_KEY, "start", ex=60)
    logger.info("已设置启动信号")


def run_voice_process(config_dict, paths_dict, video_path, run_id):
    """语音模块进程"""
    import redis
    from core.message_bus import MessageBus
    from core.frontend_sync import FrontendSync
    from core.path_manager import PathConfig
    from modules.voice import VoiceModule

    logger = setup_logger("process.voice")
    logger.info("语音进程启动")

    # 重建对象（字符串转 Path）
    from pathlib import Path
    paths = PathConfig(
        base_dir=Path(paths_dict["base_dir"]),
        data_root=Path(paths_dict["data_root"]),
        model_root=Path(paths_dict["model_root"]),
        result_root=Path(paths_dict["result_root"]),
    )
    bus = MessageBus(redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
                   consumer_name="voice_process")
    aggregator = FrontendSync(
        fps=config_dict.get("fps", 30),
        expected_modules={"mot", "voice"},
        redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
        writer_only=True,  # 模块进程只写入，不运行聚合循环
    )
    aggregator._module_timeout = 60.0

    bus.start()
    aggregator.start()

    # 等待用户点击"开始测试"
    wait_for_start_signal(REDIS_HOST, REDIS_PORT, REDIS_DB)

    module = VoiceModule(bus, config_dict, paths, aggregator)
    module.start(video_path, run_id)

    aggregator.stop()
    bus.stop()
    logger.info("语音进程结束")


def run_mot_process(config_dict, paths_dict, video_path, run_id, frame_queue_key):
    """MOT 模块进程"""
    import redis
    from core.message_bus import MessageBus
    from core.frontend_sync import FrontendSync
    from core.path_manager import PathConfig
    from modules.mot import MOTModule

    logger = setup_logger("process.mot")
    logger.info("MOT 进程启动")

    from pathlib import Path
    paths = PathConfig(
        base_dir=Path(paths_dict["base_dir"]),
        data_root=Path(paths_dict["data_root"]),
        model_root=Path(paths_dict["model_root"]),
        result_root=Path(paths_dict["result_root"]),
    )
    bus = MessageBus(redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
                   consumer_name="mot_process")
    aggregator = FrontendSync(
        fps=config_dict.get("fps", 30),
        expected_modules={"mot", "voice"},
        redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
        writer_only=True,  # 模块进程只写入，不运行聚合循环
    )
    aggregator._module_timeout = 60.0

    bus.start()
    aggregator.start()

    # 等待用户点击"开始测试"
    wait_for_start_signal(REDIS_HOST, REDIS_PORT, REDIS_DB)

    # 创建帧队列（写入 Redis List）
    config_dict["_frame_queue_redis_key"] = frame_queue_key
    module = MOTModule(bus, config_dict, paths, aggregator)
    module.start(video_path, run_id)

    aggregator.stop()
    bus.stop()
    logger.info("MOT 进程结束")


def run_behavior_process(config_dict, paths_dict, video_path, run_id, frame_queue_key):
    """行为检测模块进程"""
    import redis
    from core.message_bus import MessageBus
    from core.frontend_sync import FrontendSync
    from core.path_manager import PathConfig
    from modules.behavior import BehaviorModule

    logger = setup_logger("process.behavior")
    logger.info("行为检测进程启动")

    from pathlib import Path
    paths = PathConfig(
        base_dir=Path(paths_dict["base_dir"]),
        data_root=Path(paths_dict["data_root"]),
        model_root=Path(paths_dict["model_root"]),
        result_root=Path(paths_dict["result_root"]),
    )
    bus = MessageBus(redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
                   consumer_name="behavior_process")
    aggregator = FrontendSync(
        fps=config_dict.get("fps", 30),
        expected_modules={"mot", "voice"},
        redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
        writer_only=True,  # 模块进程只写入，不运行聚合循环
    )
    aggregator._module_timeout = 60.0

    bus.start()
    aggregator.start()

    # 等待用户点击"开始测试"
    wait_for_start_signal(REDIS_HOST, REDIS_PORT, REDIS_DB)

    config_dict["_frame_queue_redis_key"] = frame_queue_key
    module = BehaviorModule(bus, config_dict, paths, aggregator)
    module.start(video_path, run_id)

    aggregator.stop()
    bus.stop()
    logger.info("行为检测进程结束")


def run_web_process(config_dict, paths_dict, pipeline_runner_key, run_id=None):
    """Web 服务进程"""
    import redis
    from core.message_bus import MessageBus
    from core.frontend_sync import FrontendSync
    from core.path_manager import PathConfig
    from web.http_server import create_app
    import queue

    logger = setup_logger("process.web")
    logger.info("Web 进程启动")

    from pathlib import Path
    paths = PathConfig(
        base_dir=Path(paths_dict["base_dir"]),
        data_root=Path(paths_dict["data_root"]),
        model_root=Path(paths_dict["model_root"]),
        result_root=Path(paths_dict["result_root"]),
    )
    bus = MessageBus(redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
                   consumer_name="web_process")
    aggregator = FrontendSync(
        fps=config_dict.get("fps", 30),
        expected_modules={"mot", "voice"},
        redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
        writer_only=False,  # Web进程运行聚合循环，负责推送前端
    )

    bus.start()

    # 设置推送回调
    sse_handler = None

    def make_pipeline_runner():
        """创建流水线运行函数（由 Web 进程触发）"""
        def runner():
            # 设置启动信号，通知所有模块进程开始推理
            set_start_signal(REDIS_HOST, REDIS_PORT, REDIS_DB)
        return runner

    # 帧队列适配器（从 Redis List 读取）
    class RedisFrameQueue:
        def __init__(self, redis_client, key):
            self._redis = redis_client
            self._key = key

        def get(self, timeout=5):
            result = self._redis.brpop(self._key, timeout=timeout)
            if result:
                return result[1]
            return None

        def put_nowait(self, item):
            self._redis.lpush(self._key, item)

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    frame_queues = {
        "mot": RedisFrameQueue(r, "queue:frames:mot"),
        "behavior": RedisFrameQueue(r, "queue:frames:behavior"),
    }

    # 注册制度（在 Web 进程中订阅）
    from regulations.regulation_base import RegulationRegistry
    registry = RegulationRegistry()
    registry.discover()
    regulations_config = config_dict.get("regulations", {})
    for name, reg in registry._regulations.items():
        if regulations_config.get(name, False):
            reg.subscribe_events(bus)
            logger.info(f"制度 {name} 已订阅事件")
        else:
            logger.info(f"制度 {name} 已禁用")

    app = create_app(
        config=config_dict,
        bus=bus,
        registry=registry,
        paths=paths,
        pipeline_runner=make_pipeline_runner(),
        frame_queues=frame_queues,
    )

    # 创建 FlowEvaluationPipeline（流程评估编排）
    from evaluation.flow_evaluation_pipeline import FlowEvaluationPipeline
    # 使用具体的 run_id 构造结果目录
    flow_result_dir = str(paths.get_result_dir(run_id)) if run_id else str(paths.result_root)
    flow_evaluator = FlowEvaluationPipeline(
        bus=bus,
        result_dir=flow_result_dir,
        fps=config_dict.get("fps", 30),
        push_event_fn=aggregator.push_event if aggregator else None,
    )
    logger.info(f"FlowEvaluationPipeline 已创建, 结果目录: {flow_result_dir}")

    # 设置 SSE 推送
    if aggregator:
        aggregator.set_push_callback(app.sse_handler.push)
        aggregator.start()

    logger.info("启动 Flask 服务器")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

    aggregator.stop()
    bus.stop()


def main():
    """协调器：启动各模块进程"""
    # 加载配置
    config_path = args.config or os.path.join(BASE_DIR, "config.yaml")
    config = ConfigManager(config_path)
    paths = PathConfig.from_config(config.to_dict(), BASE_DIR)

    logger.info(f"FPS={config.fps}, GPU={config.gpu}")
    logger.info(f"数据目录: {paths.data_root}")
    logger.info(f"模型目录: {paths.model_root}")
    logger.info(f"结果目录: {paths.result_root}")

    # 获取视频 FPS
    import cv2
    cap = cv2.VideoCapture(config.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.fps
    cap.release()

    # 创建结果目录
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = str(paths.get_result_dir(run_id))
    for sub in ("voice", "mot", "gaze", "behavior", "qwen"):
        os.makedirs(os.path.join(result_dir, sub), exist_ok=True)
    logger.info(f"结果目录: {result_dir}")

    # 序列化配置和路径（用于传递给子进程）
    config_dict = config.to_dict()
    config_dict["fps"] = fps
    config_dict["_redis_host"] = REDIS_HOST
    config_dict["_redis_port"] = REDIS_PORT
    config_dict["_redis_db"] = REDIS_DB
    config_dict["video_path"] = config.video_path
    paths_dict = {
        "data_root": str(paths.data_root),
        "model_root": str(paths.model_root),
        "result_root": str(paths.result_root),
        "base_dir": str(paths.base_dir),
    }

    # 清理 Redis
    import redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    r.flushdb()
    r.close()

    # 启动各模块进程
    processes = []

    if config.modules.get("voice"):
        p = multiprocessing.Process(
            target=run_voice_process,
            args=(config_dict, paths_dict, config.video_path, run_id),
            name="voice", daemon=True,
        )
        p.start()
        processes.append(("voice", p))

    if config.modules.get("mot"):
        p = multiprocessing.Process(
            target=run_mot_process,
            args=(config_dict, paths_dict, config.video_path, run_id, "queue:frames:mot"),
            name="mot", daemon=True,
        )
        p.start()
        processes.append(("mot", p))

    if config.modules.get("behavior"):
        p = multiprocessing.Process(
            target=run_behavior_process,
            args=(config_dict, paths_dict, config.video_path, run_id, "queue:frames:behavior"),
            name="behavior", daemon=True,
        )
        p.start()
        processes.append(("behavior", p))

    # 启动 Web 服务进程
    p = multiprocessing.Process(
        target=run_web_process,
        args=(config_dict, paths_dict, "pipeline:command", run_id),
        name="web", daemon=True,
    )
    p.start()
    processes.append(("web", p))

    logger.info(f"已启动 {len(processes)} 个进程: {[name for name, _ in processes]}")

    # 等待所有模块进程完成
    for name, p in processes:
        if name == "web":
            continue  # Web 进程不等待，它一直运行
        p.join()
        logger.info(f"{name} 进程结束")

    # 推送完成信号
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    r.set("pipeline:status", "done", ex=60)
    r.close()

    logger.info("═══ 流水线完成 ═══")
    logger.info(f"结果目录: {result_dir}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
