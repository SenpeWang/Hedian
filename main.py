"""
核电站监护制合规检测系统 — 入口文件（多进程版）

架构:
  main.py        → 协调器（启动各模块进程）
  core/          → 核心框架（消息总线、配置、聚合器、模块基类）
  modules/       → 业务模块（语音、Tracker、注视、行为）
  rules/         → 规则层（监护制、自唱票）
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
            # 不删除信号，让所有进程都能读到
            logger.info("收到启动信号，开始推理")
            r.close()
            return
        time.sleep(0.5)


def set_start_signal(redis_host="localhost", redis_port=6379, redis_db=0):
    """设置启动信号（由 Web 进程调用）"""
    import traceback
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
    r.set(START_SIGNAL_KEY, "start", ex=60)
    logger.info("已设置启动信号")
    logger.info("已设置启动信号")
    import traceback
    stack = traceback.format_stack()
    logger.info("调用栈: " + repr(stack[-3]))

def run_voice_process(config_dict, paths_dict, video_path, run_id):
    """语音模块进程"""
    import redis
    from core.event_bus import EventBus
    from core.display_buffer import DisplayBuffer
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
    event_bus = EventBus(redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
                   consumer_name="voice_process")
    display_buffer = DisplayBuffer(
        fps=config_dict.get("fps", 30),
        expected_modules={"tracker", "voice", "gaze"},
        redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
        writer_only=True,  # 模块进程只写入，不运行聚合循环
    )
    display_buffer._module_timeout = 60.0

    event_bus.start()
    display_buffer.start()

    # 等待用户点击"开始测试"
    wait_for_start_signal(REDIS_HOST, REDIS_PORT, REDIS_DB)

    module = VoiceModule(event_bus, config_dict, paths, display_buffer)
    module.start(video_path, run_id)

    display_buffer.stop()
    event_bus.stop()
    logger.info("语音进程结束")


def run_tracker_process(config_dict, paths_dict, video_path, run_id, frame_queue_key):
    """Tracker 模块进程"""
    import os
    import redis
    from core.event_bus import EventBus
    from core.display_buffer import DisplayBuffer
    from core.path_manager import PathConfig
    # 清除 LD_LIBRARY_PATH，让 PyTorch 使用自带的 CUDA 库
    if "LD_LIBRARY_PATH" in os.environ:
        del os.environ["LD_LIBRARY_PATH"]

    from modules.tracker import TrackerModule

    logger = setup_logger("process.tracker")
    logger.info("Tracker 进程启动")

    from pathlib import Path
    paths = PathConfig(
        base_dir=Path(paths_dict["base_dir"]),
        data_root=Path(paths_dict["data_root"]),
        model_root=Path(paths_dict["model_root"]),
        result_root=Path(paths_dict["result_root"]),
    )
    event_bus = EventBus(redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
                   consumer_name="tracker_process")
    display_buffer = DisplayBuffer(
        fps=config_dict.get("fps", 30),
        expected_modules={"tracker", "voice", "gaze"},
        redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
        writer_only=True,
    )
    display_buffer._module_timeout = 60.0

    event_bus.start()
    display_buffer.start()

    # 等待用户点击"开始测试"
    wait_for_start_signal(REDIS_HOST, REDIS_PORT, REDIS_DB)

    # 创建帧队列（写入 Redis List）
    config_dict["_frame_queue_redis_key"] = frame_queue_key
    module = TrackerModule(event_bus, config_dict, paths, display_buffer)
    module.start(video_path, run_id)

    display_buffer.stop()
    event_bus.stop()
    logger.info("Tracker 进程结束")

def run_behavior_process(config_dict, paths_dict, video_path, run_id, frame_queue_key):
    """行为检测模块进程"""
    import redis
    from core.event_bus import EventBus
    from core.display_buffer import DisplayBuffer
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
    event_bus = EventBus(redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
                   consumer_name="behavior_process")
    display_buffer = DisplayBuffer(
        fps=config_dict.get("fps", 30),
        expected_modules={"tracker", "voice", "gaze"},
        redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
        writer_only=True,  # 模块进程只写入，不运行聚合循环
    )
    display_buffer._module_timeout = 60.0

    event_bus.start()
    display_buffer.start()

    # 等待用户点击"开始测试"
    wait_for_start_signal(REDIS_HOST, REDIS_PORT, REDIS_DB)

    config_dict["_frame_queue_redis_key"] = frame_queue_key
    module = BehaviorModule(event_bus, config_dict, paths, display_buffer)
    module.start(video_path, run_id)

    display_buffer.stop()
    event_bus.stop()
    logger.info("行为检测进程结束")




def run_web_process(config_dict, paths_dict, pipeline_runner_key, run_id=None):
    """Web 服务进程"""
    import redis
    from core.event_bus import EventBus
    from core.display_buffer import DisplayBuffer
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
    event_bus = EventBus(redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
                   consumer_name="web_process")
    display_buffer = DisplayBuffer(
        fps=config_dict.get("fps", 30),
        expected_modules={"tracker", "voice", "gaze"},
        redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
        writer_only=False,
    )

    sse_handler = None

    def make_pipeline_runner():
        def runner():
            set_start_signal(REDIS_HOST, REDIS_PORT, REDIS_DB)
        return runner

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
        "tracker": RedisFrameQueue(r, "queue:frames:tracker"),
        "behavior": RedisFrameQueue(r, "queue:frames:behavior"),
    }

    from rules.rule_base import RuleRegistry
    registry = RuleRegistry()
    registry.discover()
    rules_config = config_dict.get("rules", {})
    for name, reg in registry._rules.items():
        if rules_config.get(name, False):
            reg.subscribe_events(event_bus)
            logger.info(f"制度 {name} 已订阅事件")
        else:
            logger.info(f"制度 {name} 已禁用")

    event_bus.start()

    app = create_app(
        config=config_dict,
        event_bus=event_bus,
        registry=registry,
        paths=paths,
        pipeline_runner=make_pipeline_runner(),
        frame_queues=frame_queues,
    )

    from evaluation.flow_evaluation_manager import FlowEvaluationManager
    flow_result_dir = str(paths.get_result_dir(run_id)) if run_id else str(paths.result_root)
    flow_evaluator = FlowEvaluationManager(
        event_bus=event_bus,
        result_dir=flow_result_dir,
        fps=config_dict.get("fps", 30),
        display_fn=display_buffer.push_display if display_buffer else None,
    )
    logger.info(f"FlowEvaluationManager 已创建, 结果目录: {flow_result_dir}")

    if display_buffer:
        def push_wrapper(event):
            if event is None or (isinstance(event, dict) and event.get("type") == "done"):
                app.pipeline_state["status"] = "idle"
                logger.info("检测到流水线运行结束信号，已将 Web 状态置为 idle")
                # 保存规则事件到JSON
                if flow_result_dir:
                    registry.save_all_results(flow_result_dir)
            app.sse_handler.push(event)
        display_buffer.set_push_callback(push_wrapper)
        display_buffer.start()

    logger.info("启动 Flask 服务器")
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=True)

    display_buffer.stop()
    event_bus.stop()


def main():
    """协调器：启动各模块进程"""
    config_path = args.config or os.path.join(BASE_DIR, "config.yaml")
    config = ConfigManager(config_path)
    paths = PathConfig.from_config(config.to_dict(), BASE_DIR)

    logger.info(f"FPS={config.fps}, GPU={config.gpu}")
    logger.info(f"数据目录: {paths.data_root}")
    logger.info(f"模型目录: {paths.model_root}")
    logger.info(f"结果目录: {paths.result_root}")

    import cv2
    cap = cv2.VideoCapture(config.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.fps
    cap.release()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = str(paths.get_result_dir(run_id))
    for sub in ("voice", "tracker", "gaze", "behavior", "qwen"):
        os.makedirs(os.path.join(result_dir, sub), exist_ok=True)
    logger.info(f"结果目录: {result_dir}")

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

    import redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    r.flushdb()
    r.close()

    r2 = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    r2.delete(START_SIGNAL_KEY)
    r2.close()
    logger.info("已清除启动信号，等待用户点击'开始测试'")

    r3 = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    for key in r3.scan_iter("gaze:*"):
        r3.delete(key)
    for key in r3.scan_iter("inference:*"):
        r3.delete(key)
    for key in r3.scan_iter("module:*"):
        r3.delete(key)
    for key in r3.scan_iter("pipeline:*"):
        r3.delete(key)
    r3.close()
    logger.info("已清理 Redis 缓存")

    processes = []

    if config.modules.get("voice"):
        p = multiprocessing.Process(
            target=run_voice_process,
            args=(config_dict, paths_dict, config.video_path, run_id),
            name="voice", daemon=True,
        )
        p.start()
        processes.append(("voice", p))

    if config.modules.get("tracker"):
        p = multiprocessing.Process(
            target=run_tracker_process,
            args=(config_dict, paths_dict, config.video_path, run_id, "queue:frames:tracker"),
            name="tracker", daemon=True,
        )
        p.start()
        processes.append(("tracker", p))

    if config.modules.get("behavior"):
        p = multiprocessing.Process(
            target=run_behavior_process,
            args=(config_dict, paths_dict, config.video_path, run_id, "queue:frames:behavior"),
            name="behavior", daemon=True,
        )
        p.start()
        processes.append(("behavior", p))

    # 凝视估计由 tracker 模块内部调用（独立实现，非独立进程）

    p = multiprocessing.Process(
        target=run_web_process,
        args=(config_dict, paths_dict, "pipeline:command", run_id),
        name="web", daemon=True,
    )
    p.start()
    processes.append(("web", p))

    logger.info(f"已启动 {len(processes)} 个进程: {[name for name, _ in processes]}")

    # 只等 web 进程（模块推理完成由 ModuleSync 自主检测）
    web_p = processes[-1][1] if processes else None
    if web_p:
        web_p.join()
        logger.info("Web 进程结束")

    logger.info("═══ 流水线完成 ═══")
    logger.info(f"结果目录: {result_dir}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
