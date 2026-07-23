"""
核电站监护制合规检测系统 — 入口文件

架构:
  main.py        → 协调器，启动各模块进程
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

# 设置 GPU (主进程和 Web 进程不限制可见性，保证 Web 可以调用物理 GPU 0 运行 Qwen)
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# 导入核心模块
from core.config_manager import ConfigManager
from core.path_manager import PathConfig
from core.logger import setup_logger, add_root_file_handler

# Redis 配置
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

logger = setup_logger("main")

# 启动信号 Redis key
START_SIGNAL_KEY = "pipeline:start_signal"


def wait_for_start_signal(redis_host="localhost", redis_port=6379, redis_db=0, last_signal=None):
    """等待用户点击'开始测试'的信号，代际机制确保只响应新信号

    Args:
        last_signal: 上一次消费的信号值，仅当 Redis 中的信号值不同（且非空）时返回

    Returns:
        新的信号值（时间戳字符串）
    """
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
    logger.info(f"等待用户点击'开始测试'... (last_signal={last_signal})")
    while True:
        signal = r.get(START_SIGNAL_KEY)
        if signal and signal != last_signal:
            logger.info(f"收到启动信号，开始推理 (signal={signal})")
            r.close()
            return signal
        time.sleep(0.5)


def set_start_signal(redis_host="localhost", redis_port=6379, redis_db=0):
    """设置启动信号（由 Web 进程调用），使用时间戳作为代际标识，与 /start 路由保持一致"""
    import time as _time
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
    r.set(START_SIGNAL_KEY, str(_time.time()), ex=3600)
    logger.info("已设置启动信号")

def _run_module_process(
    module_name: str,
    module_factory,
    config_dict: dict,
    paths_dict: dict,
    video_path: str,
    run_id: str,
    *,
    env_setup=None,
):
    """业务模块进程的通用模板

    Args:
        module_name: 模块名称，用于日志和 EventBus 消费者名
        module_factory: 接收 (event_bus, config, paths, display_buffer) 返回模块实例的可调用对象
        env_setup: 可选的环境变量预处理回调（如清除 LD_LIBRARY_PATH）
    """
    # 限制业务子模块仅可见指定的推理 GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = config_dict.get("gpu", "1")
    import redis
    from pathlib import Path
    from core.event_bus import EventBus
    from core.inference_bus import InferenceBus
    from core.path_manager import PathConfig

    if env_setup:
        env_setup()

    _log_file = config_dict.get("_log_file")
    if _log_file:
        add_root_file_handler(_log_file)

    logger = setup_logger(f"process.{module_name}")
    logger.info(f"{module_name} 进程启动")

    paths = PathConfig(
        base_dir=Path(paths_dict["base_dir"]),
        data_root=Path(paths_dict["data_root"]),
        model_root=Path(paths_dict["model_root"]),
        result_root=Path(paths_dict["result_root"]),
    )

    _r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    last_signal = _r.get(START_SIGNAL_KEY)
    _r.close()
    logger.info(f"{module_name} 进程启动，当前信号={last_signal}，等待用户点击'开始测试'触发新推理")

    while True:
        last_signal = wait_for_start_signal(REDIS_HOST, REDIS_PORT, REDIS_DB, last_signal)
        
        # 动态解析信号代际生成全新的批次运行ID，彻底隔离每次运行的结果目录！
        try:
            sig_time = float(last_signal)
            active_run_id = datetime.fromtimestamp(sig_time).strftime("%Y%m%d_%H%M%S")
        except Exception:
            active_run_id = run_id
            
        logger.info(f"{module_name} 进程开始新一轮推理 run_id={active_run_id}")
        
        # 为全新的动态 run_id 自动创建结果子目录结构，防止路径缺失异常
        active_result_dir = str(paths.get_result_dir(active_run_id))
        for sub in ("voice", "tracker", "gaze", "behavior", "qwen", "evaluation"):
            os.makedirs(os.path.join(active_result_dir, sub), exist_ok=True)
            
        # 动态将当前子进程的日志重定向到当前活跃 Session 文件夹下的日志文件
        new_log_file = os.path.join(active_result_dir, "run.log")
        from core.logger import redirect_file_logger
        redirect_file_logger(new_log_file)
            
        event_bus = EventBus(
            redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
            consumer_name=f"{module_name}_process",
        )
        display_buffer = InferenceBus(
            fps=config_dict.get("fps", 30),
            redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
        )
        event_bus.start()
        display_buffer.start()

        module = module_factory(event_bus, config_dict, paths, display_buffer)
        module.start(video_path, active_run_id)

        display_buffer.stop()
        event_bus.stop()
        logger.info(f"{module_name} 进程完成本轮推理 run_id={run_id}，等待下一次触发")


def run_voice_process(config_dict, paths_dict, video_path, run_id):
    """语音模块进程"""
    from modules.voice import VoiceModule
    _run_module_process(
        "voice", lambda *a: VoiceModule(*a), config_dict, paths_dict, video_path, run_id,
    )


def run_tracker_process(config_dict, paths_dict, video_path, run_id):
    """Tracker 模块进程"""
    import os
    # 清除 LD_LIBRARY_PATH，让 PyTorch 使用自带的 CUDA 库
    def _env_setup():
        if "LD_LIBRARY_PATH" in os.environ:
            del os.environ["LD_LIBRARY_PATH"]
    from modules.tracker import TrackerModule
    _run_module_process(
        "tracker", lambda *a: TrackerModule(*a), config_dict, paths_dict, video_path, run_id,
        env_setup=_env_setup,
    )


def run_behavior_process(config_dict, paths_dict, video_path, run_id):
    """行为检测模块进程"""
    from modules.behavior import BehaviorModule
    _run_module_process(
        "behavior", lambda *a: BehaviorModule(*a), config_dict, paths_dict, video_path, run_id,
    )


def run_web_process(config_dict, paths_dict, pipeline_runner_key, run_id=None):
    """Web 服务进程"""
    import redis
    from pathlib import Path
    from core.event_bus import EventBus
    from core.module_sync import ModuleSync
    from core.path_manager import PathConfig
    from web.http_server import create_app

    # 子进程需重新挂载 root 文件 handler
    _log_file = config_dict.get("_log_file")
    if _log_file:
        add_root_file_handler(_log_file)

    logger = setup_logger("process.web")
    logger.info("Web 进程启动")
    config_dict["run_id"] = run_id

    paths = PathConfig(
        base_dir=Path(paths_dict["base_dir"]),
        data_root=Path(paths_dict["data_root"]),
        model_root=Path(paths_dict["model_root"]),
        result_root=Path(paths_dict["result_root"]),
    )
    event_bus = EventBus(redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
                   consumer_name="web_process")
    import cv2
    video_duration = 0.0
    try:
        cap = cv2.VideoCapture(config_dict.get("video_path"))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or config_dict.get("fps", 30)
        video_duration = total_frames / video_fps if video_fps > 0 else 0.0
        cap.release()
        logger.info(f"读取视频时长完成: {video_duration:.2f}s")
    except Exception as e:
        logger.error(f"读取视频时长失败: {e}", exc_info=True)

    display_buffer = ModuleSync(
        fps=config_dict.get("fps", 30),
        expected_modules={"voice", "tracker", "gaze", "behavior"},
        redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
        duration=video_duration,
    )

    def make_pipeline_runner():
        def runner():
            set_start_signal(REDIS_HOST, REDIS_PORT, REDIS_DB)
        return runner

    # 结果目录提前确定，供规则和评估器使用
    flow_result_dir = str(paths.get_result_dir(run_id)) if run_id else str(paths.result_root)
    flow_evaluator = None # 提前声明，供动态订阅回调在运行时直接闭包捕捉

    from rules.rule_base import RuleRegistry
    from rules.flow_recorder import FlowEventRecorder
    registry = RuleRegistry()
    registry.discover()

    # 流程事件记录器：统一保存 rules/flow_events.json
    flow_recorder = FlowEventRecorder(event_bus)
    flow_recorder.set_result_dir(flow_result_dir)
    logger.info(f"FlowEventRecorder 已创建, 结果目录: {flow_result_dir}")

    rules_config = config_dict.get("rules", {})
    for name, reg in registry._rules.items():
        if rules_config.get(name, False):
            # 传入结果目录
            if hasattr(reg, "set_result_dir"):
                reg.set_result_dir(flow_result_dir)
            reg.subscribe_events(event_bus)
            logger.info(f"制度 {name} 已订阅事件")
        else:
            logger.info(f"制度 {name} 已禁用")

    # 定义 pipeline 启动通知回调，以动态重置所有数据记录器和规则的结果路径目录
    def on_pipeline_start(msg):
        nonlocal flow_result_dir
        data = msg.get("data", {})
        active_run_id = data.get("run_id")
        if not active_run_id:
            return
        active_result_dir = str(paths.get_result_dir(active_run_id))
        flow_result_dir = active_result_dir
        config_dict["run_id"] = active_run_id
        
        # 动态将当前 Web 进程的日志重定向到当前活跃 Session 文件夹下的日志文件
        new_log_file = os.path.join(active_result_dir, "run.log")
        from core.logger import redirect_file_logger
        redirect_file_logger(new_log_file)
        
        flow_recorder.set_result_dir(active_result_dir)
        for rname, rreg in registry._rules.items():
            if rules_config.get(rname, False):
                if hasattr(rreg, "set_result_dir"):
                    rreg.set_result_dir(active_result_dir)
        
        if flow_evaluator:
            flow_evaluator.set_result_dir(active_result_dir)
            
    event_bus.subscribe("pipeline.start", on_pipeline_start)
    event_bus.start()

    app = create_app(
        config=config_dict,
        event_bus=event_bus,
        registry=registry,
        paths=paths,
        pipeline_runner=make_pipeline_runner(),
    )

    # 大模型评估结果和流式推理文本：具备高实时性，不应随视频播放进度被拖延（对齐）
    # 定义直接推送函数，使评测卡片在生成过程中能够实时显示在前端
    # 但必须等该流程在前端播放结束后（即全局时钟追赶上流程结束时间）才允许推送
    def push_to_sse_directly(event_type: str, data: dict) -> None:
        if event_type in ("segment_report", "segment_report_stream"):
            local_sec = data.get("localSec", 0.0)
            
            # 如果 display_buffer 还在播放该流程之前的内容，则阻塞等待，直到前端视频进度追赶上流程的结束时间
            if display_buffer and local_sec > 0:
                while display_buffer._pushed_global_sec < local_sec:
                    # 如果系统状态已经变为 idle（说明流程结束或用户停止测试），立刻放行，防止死锁
                    if app.pipeline_state.get("status") == "idle":
                        break
                    time.sleep(0.1)
                    
            batch = {
                "globalSec": 0.0,
                event_type: [
                    {
                        "localSec": data.get("localSec"),
                        "tag": data.get("tag"),
                        "data": data.get("data")
                    }
                ]
            }
            app.sse_handler.push(batch)
        else:
            if display_buffer:
                display_buffer.push_display(event_type, data)

    from evaluation.flow_evaluation_manager import FlowEvaluationManager
    flow_evaluator = FlowEvaluationManager(
        event_bus=event_bus,
        result_dir=flow_result_dir,
        fps=config_dict.get("fps", 30),
        display_fn=push_to_sse_directly,
    )
    logger.info(f"FlowEvaluationManager 已创建, 结果目录: {flow_result_dir}")

    if display_buffer:
        def push_wrapper(event):
            # 推理流结束信号：ModuleSync 推送 None 表示流水线完成
            if event is None or (isinstance(event, dict) and event.get("source") == "done"):
                app.pipeline_state["status"] = "idle"
                logger.info("检测到流水线运行结束信号，已将 Web 状态置为 idle")
                # 1. 收尾所有制度：finalize 关闭活跃流程(触发 FLOW_ENDED) + 保存制度事件
                if flow_result_dir:
                    registry.save_all_results(flow_result_dir)
                # 2. 收尾评估器：处理未触发 FLOW_ENDED 的活跃流程，并等待所有评估任务完成
                #    评估完成后 segment_report/segment_report_stream 会写入 Redis Stream
                try:
                    flow_evaluator.finalize()
                except Exception as e:
                    logger.error(f"FlowEvaluationManager finalize 失败: {e}", exc_info=True)
                # 3. 刷新 Stream 中剩余事件（评估结果等）到前端，确保 done 之前不丢数据
                try:
                    display_buffer.flush_remaining()
                except Exception as e:
                    logger.error(f"刷新剩余事件失败: {e}", exc_info=True)
            app.sse_handler.push(event)
            # 一次性运行：done 送达后（此时评估已全部完成、结果已落盘）终止所有模块子进程并退出 Web。
            # 模块子进程即 GPU 占用进程，等价于训练结束后 kill GPU 进程，确保 GPU 显存随之释放；
            # main 的监控循环检测到 web 退出后会自然收尾退出。
            if event is None:
                import threading

                def _terminate_pipeline():
                    time.sleep(2.0)  # 确保 SSE 已把 done 送达前端
                    logger.info("推理结果已全部展示完毕，终止所有模块子进程并退出 Web（释放 GPU）")
                    try:
                        import subprocess
                        # [p] 方括号防止 pkill 匹配到自身命令行
                        subprocess.run(["pkill", "-f", "main.py --g[p]u"], timeout=10)
                    except Exception as e:
                        logger.error(f"终止模块子进程失败: {e}")
                    time.sleep(0.5)
                    os._exit(0)

                threading.Thread(target=_terminate_pipeline, daemon=True).start()
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

    # 日志文件写入 result_dir 下的 run.log
    log_file = os.path.join(result_dir, "run.log")
    add_root_file_handler(log_file)
    logger.info(f"日志文件: {log_file}")

    config_dict = config.to_dict()
    config_dict["fps"] = fps
    config_dict["gpu"] = args.gpu
    config_dict["_redis_host"] = REDIS_HOST
    config_dict["_redis_port"] = REDIS_PORT
    config_dict["_redis_db"] = REDIS_DB
    config_dict["_log_file"] = log_file
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
            name="voice", daemon=False,
        )
        p.start()
        processes.append(("voice", p))

    if config.modules.get("tracker"):
        p = multiprocessing.Process(
            target=run_tracker_process,
            args=(config_dict, paths_dict, config.video_path, run_id),
            name="tracker", daemon=False,
        )
        p.start()
        processes.append(("tracker", p))

    if config.modules.get("behavior"):
        p = multiprocessing.Process(
            target=run_behavior_process,
            args=(config_dict, paths_dict, config.video_path, run_id),
            name="behavior", daemon=False,
        )
        p.start()
        processes.append(("behavior", p))

    # 凝视估计由 tracker 模块内部调用（独立实现，非独立进程）

    p = multiprocessing.Process(
        target=run_web_process,
        args=(config_dict, paths_dict, "pipeline:command", run_id),
        name="web", daemon=False,
    )
    p.start()
    processes.append(("web", p))

    logger.info(f"已启动 {len(processes)} 个进程: {[name for name, _ in processes]}")

    # 监控所有子进程：模块进程异常退出时记录告警（便于定位），web 进程退出则收尾
    module_procs = [(name, p) for name, p in processes if name != "web"]
    web_p = processes[-1][1] if processes and processes[-1][0] == "web" else None
    crashed = set()
    while True:
        for name, p in module_procs:
            if name in crashed:
                continue
            if not p.is_alive():
                code = p.exitcode
                if code not in (0, None):
                    logger.error(f"模块进程 [{name}] 异常退出 (exitcode={code})，该路数据将停更；请检查对应日志")
                else:
                    logger.info(f"模块进程 [{name}] 已正常退出 (exitcode={code})")
                crashed.add(name)
        if web_p is None or not web_p.is_alive():
            logger.info("Web 进程结束")
            break
        time.sleep(1.0)

    logger.info("═══ 流水线完成 ═══")
    logger.info(f"结果目录: {result_dir}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
