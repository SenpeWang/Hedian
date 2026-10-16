"""核电站监护制合规检测系统入口文件.

架构:
  main.py        -> 协调器，启动各模块进程
  core/          -> 核心框架（消息总线、配置、聚合器、模块基类）
  modules/       -> 业务模块（语音、Tracker、注视、行为）
  rules/         -> 规则层（监护制、信息通报、人员状态）
  evaluation/    -> 评估层（规则评估、大模型评估）
  web/           -> 前端层（HTTP 服务器、WebSocket）

多进程架构:
  每个模块运行在独立进程中，通过 Redis 通信。
"""
import sys
import os
import glob

# 让 onnxruntime-gpu 的 CUDA EP 能找到 pip 装的 nvidia cu12 运行时库
# (库在 site-packages/nvidia/*/lib, 不在系统标准路径, 动态链接器默认不搜;
#  设进 LD_LIBRARY_PATH 后 spawn 的子进程继承之, gaze 的 ONNX 才上 GPU 而非 fallback CPU)
try:
    _nv_root = os.path.join(sys.prefix, "lib", "python%d.%d" % sys.version_info[:2], "site-packages", "nvidia")
    if not os.path.isdir(_nv_root):
        import site
        for _site_pkg in site.getsitepackages():
            _candidate = os.path.join(_site_pkg, "nvidia")
            if os.path.isdir(_candidate):
                _nv_root = _candidate; break
    _nv_libs = glob.glob(os.path.join(_nv_root, "*", "lib"))
    if _nv_libs:
        _current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(_nv_libs) + (":" + _current if _current else "")
except Exception:
    pass

import argparse
import multiprocessing
import threading
import redis
import cv2
from datetime import datetime

parser = argparse.ArgumentParser(description="核电站监护制合规检测系统")
parser.add_argument("--gpu", type=str, default="0", help="GPU 编号 (默认: 0)")
parser.add_argument("--config", type=str, default=None, help="配置文件路径")
args = parser.parse_args()

# 视角级 GPU 分配: 业务子进程经 _run_module_process 按 module_name 取 gpu_map; 评估子进程经 eval_gpu 分配
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
# 切到项目根，使配置路径以相对形式表达（models/...、data/...）且不依赖启动位置
os.chdir(BASE_DIR)

from core.config_manager import ConfigManager
from core.path_manager import PathConfig
from core.logger import setup_logger, add_root_file_handler

def _resolve_redis_conf():
    """解析 Redis 连接参数: config.yaml 的 redis 段 > 内置默认值, 环境变量优先级最高.

    必须在模块导入期完成解析: multiprocessing spawn 出的推理子进程会重新导入本模块,
    只有模块级常量才能保证 Web 进程与所有推理子进程连接到同一实例/库.

    Returns:
        (host, port, db) 三元组.
    """
    host, port, db = "localhost", 6379, 0
    cfg_path = args.config or os.path.join(BASE_DIR, "config.yaml")
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as conf_file:
            redis_conf = (yaml.safe_load(conf_file) or {}).get("redis") or {}
        host = str(redis_conf.get("host", host))
        port = int(redis_conf.get("port", port))
        db = int(redis_conf.get("db", db))
    except Exception as error:
        # 配置缺失/损坏时沿用默认值, 不阻断启动
        logger.warning(f"读取配置 redis 段失败, 沿用默认 {host}:{port}/{db}: {error}")

    # 同机多实例共用 Redis 实例时, 用环境变量为当前实例指定独立 db/端口实现隔离
    host = os.environ.get("HEDIAN_REDIS_HOST", host)
    port = int(os.environ.get("HEDIAN_REDIS_PORT", port))
    db = int(os.environ.get("HEDIAN_REDIS_DB", db))
    return host, port, db


logger = setup_logger("main")
REDIS_HOST, REDIS_PORT, REDIS_DB = _resolve_redis_conf()
logger.info(f"Redis 目标: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")


def _paths_from_dict(paths_dict: dict) -> "PathConfig":
    """从可序列化的 paths_dict 重建 PathConfig（跨进程传递用）.

    Args:
        paths_dict: 含 base_dir / data_root / model_root / result_root 的可 pickle 字典.

    Returns:
        重建的 PathConfig 实例.
    """
    from pathlib import Path
    return PathConfig(
        base_dir=Path(paths_dict["base_dir"]),
        data_root=Path(paths_dict["data_root"]),
        model_root=Path(paths_dict["model_root"]),
        result_root=Path(paths_dict["result_root"]),
    )


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
    """业务模块进程的通用模板（单次推理，完成后自动退出释放 VRAM）.

    Args:
        module_name: 模块名, 用于取 gpu_map 与日志标识.
        module_factory: 模块构造函数(如 VoiceModule).
        config_dict: 完整配置字典.
        paths_dict: 可序列化的路径配置字典.
        video_path: 待推理视频路径.
        run_id: 本轮运行标识.
        env_setup: 可选的环境变量修正回调, 在导入模块前执行.
    """
    # 业务子模块仅可见指定的推理 GPU
    # 视角级GPU: gpu_map 按模块名取, 未配置则回退 gpu_default(已含CLI --gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config_dict.get("gpu_map", {}).get(module_name, config_dict.get("gpu_default", "0")))
    # 硬性要求: 仅允许 GPU 推理; CUDA 初始化失败立即快速失败, 严禁静默回退 CPU
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"[{module_name}] CUDA 初始化失败(CUDA_VISIBLE_DEVICES="
            f"{os.environ['CUDA_VISIBLE_DEVICES']}), "
            "按约定禁止 CPU 回退, 拒绝启动"
        )
    from pathlib import Path
    from core.event_bus import EventBus
    from core.inference_stream import InferenceStream
    from core.path_manager import PathConfig

    if env_setup:
        env_setup()

    _log_file = config_dict.get("_log_file")
    if _log_file:
        add_root_file_handler(_log_file)

    logger = setup_logger(f"process.{module_name}")
    logger.info(f"{module_name} 进程启动，run_id={run_id}")

    paths = PathConfig(
        base_dir=Path(paths_dict["base_dir"]),
        data_root=Path(paths_dict["data_root"]),
        model_root=Path(paths_dict["model_root"]),
        result_root=Path(paths_dict["result_root"]),
    )

    active_result_dir = str(paths.get_result_dir(run_id))
    for sub in ("voice", "tracker", "gaze", "behavior", "qwen", "evaluation"):
        os.makedirs(os.path.join(active_result_dir, sub), exist_ok=True)

    new_log_file = os.path.join(active_result_dir, "run.log")
    from core.logger import redirect_file_logger
    redirect_file_logger(new_log_file)

    event_bus = EventBus(
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        redis_db=REDIS_DB,
        consumer_name=f"{module_name}_process",
    )
    inference_stream = InferenceStream(
        fps=config_dict.get("fps", 30),
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        redis_db=REDIS_DB,
    )
    event_bus.start()
    inference_stream.start()

    module = module_factory(event_bus, config_dict, paths, inference_stream)
    module.start(video_path, run_id)

    inference_stream.stop()
    event_bus.stop()
    logger.info(
        f"{module_name} 进程完成本轮推理 run_id={run_id}，顺利退出并 100% 释放 GPU VRAM")


def run_voice_process(config_dict, paths_dict, video_path, run_id):
    """运行语音处理进程.

    Args:
        config_dict: 完整配置字典.
        paths_dict: 可序列化的路径配置字典.
        video_path: 待推理视频路径.
        run_id: 本轮运行标识.
    """
    from modules.voice import VoiceModule
    _run_module_process("voice", VoiceModule, config_dict, paths_dict, video_path, run_id)


def run_tracker_process(config_dict, paths_dict, video_path, run_id):
    """运行跟踪器处理进程.

    Args:
        config_dict: 完整配置字典.
        paths_dict: 可序列化的路径配置字典.
        video_path: 待推理视频路径.
        run_id: 本轮运行标识.
    """
    # 清除 LD_LIBRARY_PATH，让 PyTorch 使用自带的 CUDA 库
    def _env_setup():
        """清理 LD_LIBRARY_PATH, 避免 nvidia pip 运行时库被 PyTorch 误链接."""
        if "LD_LIBRARY_PATH" in os.environ:
            del os.environ["LD_LIBRARY_PATH"]

    from modules.tracker import TrackerModule
    _run_module_process("tracker", TrackerModule, config_dict, paths_dict,
                        video_path, run_id, env_setup=_env_setup)


def run_behavior_process(config_dict, paths_dict, video_path, run_id):
    """运行行为处理进程.

    Args:
        config_dict: 完整配置字典.
        paths_dict: 可序列化的路径配置字典.
        video_path: 待推理视频路径.
        run_id: 本轮运行标识.
    """
    from modules.behavior import BehaviorModule
    _run_module_process("behavior", BehaviorModule, config_dict, paths_dict,
                        video_path, run_id)


def run_web_process(config_dict, paths_dict, run_id=None):
    """运行 web 进程.

    启动事件总线、推理同步器、规则与评估器, 并按需拉起推理子进程,
    最后阻塞在 uvicorn 上.

    Args:
        config_dict: 完整配置字典, 运行中会被写入 run_id / duration 等键.
        paths_dict: 可序列化的路径配置字典.
        run_id: 初始运行标识; None 时规则与评估器使用 result_root 级目录.
    """
    from pathlib import Path
    from core.event_bus import EventBus
    from core.inference_sync import InferenceSync
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
    event_bus = EventBus(redis_host=REDIS_HOST,
                            redis_port=REDIS_PORT,
                            redis_db=REDIS_DB,
                            consumer_name="web_process")
    video_duration = 0.0
    try:
        # 多视角视频帧率/时长可能不同（front 30fps / pop 25fps），
        # 取所有视频的最大时长作为对齐 duration，避免较短视频提前剔除仍活跃的 source
        video_paths = []
        for view_key in ("front", "pop"):
            video_path = config_dict.get("videos", {}).get(view_key)
            if video_path:
                video_paths.append(video_path)
        if not video_paths and config_dict.get("video_path"):
            video_paths.append(config_dict["video_path"])
        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS) or config_dict.get("fps", 30)
            if total_frames > 0 and video_fps > 0:
                duration = total_frames / video_fps
                video_duration = max(video_duration, duration)
                logger.info(
                    f"视频 {video_path}: {total_frames}帧@{video_fps:.1f}fps, 时长 {duration:.2f}s"
                )
            cap.release()
        logger.info(f"读取视频时长完成（取最大）: {video_duration:.2f}s")
    except Exception as e:
        logger.error(f"读取视频时长失败: {e}", exc_info=True)

    # 供 WSHandler connect 补发状态快照(前端刷新即拿到总时长, 不依赖前端 video 加载)
    config_dict["duration"] = video_duration

    # expected_sources 按启用的模块推导：单视角演示（如只跑 front、behavior 关闭）时必须剔除
    # behavior，否则 _all_sources_done() 永不为真 → 流水线永不收尾、结果不落盘、评估不触发
    modules_config = config_dict.get("modules", {})
    expected_sources = set()
    if modules_config.get("voice", True):
        expected_sources.add("voice")        # 语音转录
    if modules_config.get("tracker", True):
        expected_sources.update(("tracking", "gaze"))   # 目标跟踪与在岗人数 + 凝视估计
    if modules_config.get("behavior", True):
        expected_sources.add("behavior")     # 行为事件（pop 视角）
    # vis_front/vis_pop 已改为画进帧 fMP4,不再走推理流,故不参与 done 判定
    if not expected_sources:
        # 兜底：模块全关时退回全量集合，保持原有行为
        expected_sources = {"voice", "tracking", "behavior", "gaze"}

    inference_sync = InferenceSync(
        fps=config_dict.get("fps", 30),
        # per-source 粒度：每个 source 独立上报进度与结束信号，彻底解耦各模态产出长度不一致问题
        expected_sources=expected_sources,
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        redis_db=REDIS_DB,
        duration=video_duration,
        event_bus=event_bus,
    )
    # 结果目录提前确定，供规则和评估器使用
    flow_result_dir = str(paths.get_result_dir(run_id)) if run_id else str(
        paths.result_root)
    flow_evaluator = None  # 提前声明，供动态订阅回调在运行时直接闭包捕捉

    from rules.rule_base import RuleRegistry
    from rules.flow_recorder import FlowEventRecorder
    registry = RuleRegistry()
    registry.discover()

    # 流程事件记录器：统一保存 rules/flow_events.json
    flow_recorder = FlowEventRecorder(event_bus)
    flow_recorder.set_result_dir(flow_result_dir)
    logger.info(f"FlowEventRecorder 已创建, 结果目录: {flow_result_dir}")

    rules_config = config_dict.get("rules", {})
    for rule_name, rule in registry._rules.items():
        if rules_config.get(rule_name, False):
            if hasattr(rule, "set_result_dir"):
                rule.set_result_dir(flow_result_dir)
            # 同名配置段回灌制度（如 info_notice.max_flow_sec 决定超时兜底时长）
            if hasattr(rule, "apply_config"):
                rule.apply_config(config_dict.get(rule_name, {}) or {})
            rule.subscribe_events(event_bus)
            logger.info(f"制度 {rule_name} 已订阅事件")
        else:
            logger.info(f"制度 {rule_name} 已禁用")

    active_worker_processes = []

    def stop_pipeline_processes() -> int:
        """停止本轮所有推理子进程并回收其状态码.

        同时强制回收大模型评估子进程：它由 QwenEvaluator 自行 spawn，不在
        active_worker_processes 列表内，若不单独处理，其占用的约 15GB 显存
        会一直不释放（日志里那句「已 100% 清理」即为假象），下一轮推理加载
        模型时显存不足，推理子进程会静默起不来，表现为「点开始测试没反应」。

        Returns:
            实际终止的推理子进程数量（不含评估子进程）.
        """
        count = 0
        for proc_name, proc in list(active_worker_processes):
            if proc.is_alive():
                logger.info(f"终止子进程 [{proc_name}]...")
                proc.terminate()
                proc.join(timeout=1.0)
                if proc.is_alive():
                    # SIGTERM 对卡在 CUDA 上下文里的进程可能无效，升级为 SIGKILL
                    logger.warning(f"子进程 [{proc_name}] 未响应 SIGTERM，强制 kill")
                    proc.kill()
                    proc.join(timeout=1.0)
                count += 1
            else:
                proc.join(timeout=0.1)
        active_worker_processes.clear()

        # 评估子进程不在上面的列表里，单独回收，避免显存泄漏拖垮下一轮
        if flow_evaluator is not None:
            try:
                killed = flow_evaluator.terminate_workers()
                if killed:
                    logger.info(f"已强制回收 {killed} 个残留评估子进程（释放其占用的显存）")
            except Exception as kill_error:
                logger.warning(f"回收评估子进程失败: {kill_error}")

        # 触发 Linux 内核清理已结束进程的 <defunct> 僵尸表项
        multiprocessing.active_children()
        return count

    def _do_pipeline_start(event: dict) -> None:
        """按配置拉起本轮推理子进程（在工作线程内执行）.

        启动链路包含 Redis 清理、日志切换、fork 子进程等可能耗时或阻塞的操作，
        不能放在事件总线的监听线程里同步执行：监听线程一旦卡在任意一步，后续
        所有总线消息都拿不到 ACK、也不再被消费，表现为"点开始测试毫无反应，
        直到重启服务"（实测 /status 停在 running，pending 持续堆积）。

        Args:
            event: 总线消息字典, 其 "data" 字段须含 "run_id".
        """
        nonlocal flow_result_dir
        payload = event.get("data", {})
        active_run_id = payload.get("run_id")
        if not active_run_id:
            return
        active_result_dir = str(paths.get_result_dir(active_run_id))
        flow_result_dir = active_result_dir
        config_dict["run_id"] = active_run_id

        # 终止上一轮残留的推理进程并回收状态码
        stop_pipeline_processes()

        # 将 Web 进程日志重定向到当前 Session run.log
        new_log_file = os.path.join(active_result_dir, "run.log")
        from core.logger import redirect_file_logger
        redirect_file_logger(new_log_file)

        # 说明: 本轮 Redis 清理已由 http_server 的 /start（clear_redis_state）完成，
        # 此处原先再清一遍属冗余；而删除/修剪超大流（vis_stream 可达数 GB）很慢，
        # 是监听线程被卡住的常见嫌疑点，故移除，只保留下方同步器状态复位。
        inference_sync.reset()
        # 视频流:清空上轮 init 缓存,启动 fMP4 转发(与推理子进程同时起)
        if app.state.ws_handler:
            app.state.ws_handler.clear_vis_cache()
        if getattr(app.state, "vis_forwarder", None):
            app.state.vis_forwarder.stop()
            app.state.vis_forwarder.start()
        flow_recorder.reset()
        flow_recorder.set_result_dir(active_result_dir)
        for rule_name, rule in registry._rules.items():
            if rules_config.get(rule_name, False):
                rule.reset()
                if hasattr(rule, "set_result_dir"):
                    rule.set_result_dir(active_result_dir)

        if flow_evaluator:
            flow_evaluator.reset()
            flow_evaluator.set_result_dir(active_result_dir)

        # 按需启动推理子进程（完成后自动 exit(0) 退出释放 VRAM）
        front_video = config_dict.get("videos", {}).get("front") or config_dict.get("video_path")
        pop_video = config_dict.get("videos", {}).get("pop") or config_dict.get("video_path")

        if config_dict.get("modules", {}).get("voice"):
            proc = multiprocessing.Process(
                target=run_voice_process,
                args=(config_dict, paths_dict, front_video, active_run_id),
                name="voice",
                daemon=False,
            )
            proc.start()
            active_worker_processes.append(("voice", proc))

        if config_dict.get("modules", {}).get("tracker"):
            proc = multiprocessing.Process(
                target=run_tracker_process,
                args=(config_dict, paths_dict, front_video, active_run_id),
                name="tracker",
                daemon=False,
            )
            proc.start()
            active_worker_processes.append(("tracker", proc))

        if config_dict.get("modules", {}).get("behavior"):
            proc = multiprocessing.Process(
                target=run_behavior_process,
                args=(config_dict, paths_dict, pop_video, active_run_id),
                name="behavior",
                daemon=False,
            )
            proc.start()
            active_worker_processes.append(("behavior", proc))

        logger.info(
            f"已按需启动 {len(active_worker_processes)} 个推理子进程: "
            f"{[proc_name for proc_name, _ in active_worker_processes]}"
        )

    def on_pipeline_start(event: dict) -> None:
        """pipeline.start 事件入口: 立即转交工作线程后返回.

        监听线程只做"派发"这一件瞬时操作, 保证能马上 ACK 并继续消费后续消息；
        真正的启动动作在 _do_pipeline_start 里执行（见其文档说明）。

        Args:
            event: 总线消息字典.
        """
        threading.Thread(
            target=_do_pipeline_start,
            args=(event,),
            name="pipeline-start",
            daemon=True,
        ).start()

    event_bus.subscribe("pipeline.start", on_pipeline_start)
    event_bus.start()

    app = create_app(
        config=config_dict,
        event_bus=event_bus,
        paths=paths,
        inference_sync=inference_sync,
    )
    app.state.stop_pipeline = stop_pipeline_processes

    # 视频流转发器:消费 Redis 中的 fMP4 段,经 WS 二进制帧推给前端 MSE
    from web.vis_forwarder import VisStreamForwarder
    vis_forwarder = VisStreamForwarder(
        app.state.ws_handler,
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        redis_db=REDIS_DB,
    )
    app.state.vis_forwarder = vis_forwarder

    # 大模型评估结果和流式推理文本：具备高实时性，不应随视频播放进度被拖延（对齐）
    # 直接推送函数：评估结果等高实时性事件绕过 batch 对齐，直接经 WebSocket 推送前端
    # 但必须等该流程在前端播放结束后（即全局时钟追赶上流程结束时间）才允许推送
    def push_direct(event_type: str, data: dict) -> None:
        """后端评估直推函数.

        完全绕过对齐中间件, 零阻塞、零延迟直推 WebSocket 文本管道;
        无 WebSocket 连接时退回对齐中间件.

        Args:
            event_type: 事件类型, 同时作为事件的 source.
            data: 事件载荷, 取其中 localSec / tag / data 三键.
        """
        if app.state.ws_handler:
            app.state.ws_handler.push_text({
                "source": event_type,
                "localSec": data.get("localSec"),
                "tag": data.get("tag"),
                "data": data.get("data"),
            })
        else:
            inference_sync.push_display(event_type, data)

    def push_sync(event_type: str, data: dict):
        """系统通知推送.

        经过对齐中间件打包入 Batch meta, 与视频帧物理时间点同帧出屏.

        Args:
            event_type: 事件类型.
            data: 事件载荷.
        """
        inference_sync.push_display(event_type, data)

    # 评估子进程 GPU: 由 config gpu_map[evaluation] 指定, web 进程注入环境变量供 FlowEvaluationManager 读取
    os.environ["_EVAL_GPU"] = str(config_dict.get("gpu_map", {}).get("evaluation", config_dict.get("gpu_default", "1")))
    # 评估并发度：默认串行，避免多个 Qwen3-8B 子进程同卡并发加载导致 CUDA OOM
    os.environ["_EVAL_MAX_WORKERS"] = str(
        int(config_dict.get("evaluation", {}).get("max_workers", 1) or 1)
    )
    from evaluation.flow_evaluation_manager import FlowEvaluationManager
    flow_evaluator = FlowEvaluationManager(
        event_bus=event_bus,
        result_dir=flow_result_dir,
        fps=config_dict.get("fps", 30),
        sync_fn=push_sync,  # 系统通知(flow_start/flow_end) ➔ 经过对齐中间件
        direct_fn=push_direct,  # 评估报告(stream/report) ➔ 完全不经过对齐中间件
        get_playback_sec_fn=lambda: app.state.ws_handler.get_playback_sec() if app.state.ws_handler else 0.0,
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        redis_db=REDIS_DB,
        # 取证等待上限: 配置 evaluation.extract_wait_timeout_sec 可覆盖默认 600s
        extract_wait_timeout_sec=float(
            config_dict.get("evaluation", {}).get(
                "extract_wait_timeout_sec",
                FlowEvaluationManager.EXTRACT_WAIT_TIMEOUT_SEC,
            )
        ),
    )
    logger.info(f"FlowEvaluationManager 已创建, 结果目录: {flow_result_dir}")

    def push_inference_event(event):
        """推理流推送回调.

        正常时把 batch 推给前端; 收到 None 或 source=="done" 信号时收尾本轮推理
        (保存制度结果、finalize 评估器、刷新剩余事件、回收子进程、清理 Redis).

        Args:
            event: 对齐后的 batch 字典, 或 None / done 哨兵表示流水线结束.
        """
        # 推理流结束信号：InferenceSync 推送 None 表示流水线完成
        if event is None or (isinstance(event, dict)
                             and event.get("source") == "done"):
            app.state.pipeline_state["status"] = "idle"
            logger.info("检测到流水线运行结束信号，已将 Web 状态置为 idle")
            # finalize 关闭未触发 FLOW_ENDED 的活跃流程（触发 FLOW_ENDED + 保存制度事件）
            # 传入视频最大时长作为流程收尾的真实结束时刻，避免被规则 _close_flow 的默认 0
            # 吞掉导致前端在 00:00 显示"流程结束"（如 supervision 在 346s 误启后一直挂着）
            if flow_result_dir:
                registry.save_all_results(flow_result_dir,
                                          final_timestamp=video_duration)
            try:
                flow_evaluator.finalize()
            except Exception as e:
                logger.error(f"FlowEvaluationManager finalize 失败: {e}",
                             exc_info=True)
            # 刷新剩余事件（评估结果等）到前端，确保 done 之前不丢数据
            try:
                inference_sync.flush_remaining()
            except Exception as e:
                logger.error(f"刷新剩余事件失败: {e}", exc_info=True)
            # 回收退场子进程状态码，清除 ps aux 中的 <defunct> 僵尸进程条目
            stop_pipeline_processes()
            # 推理完成：保留本轮 Redis 推理数据。人员对话记录/系统通知面板是流式消费，
            # 历史只存在于 Redis 流里，清掉后推理结束再刷新页面会导致两个面板全空且
            # 无法补回。下一轮「开始测试」时 http_server 的 clear_redis_state 会统一
            # 清理，因此不会串到下一轮。
            logger.info("推理完成，保留本轮 Redis 推理数据供页面回看/刷新重放（下次开始测试时清理）")
            logger.info("推理流与评估流程已全部完成，GPU 推理子进程与僵尸表项已 100% 清理，Web 保持运行")
            # 推理结束:停止视频流转发(子进程已 finalize 编码器)
            if getattr(app.state, "vis_forwarder", None):
                app.state.vis_forwarder.stop()
        app.state.ws_handler.push(event)

    inference_sync.set_push_callback(push_inference_event)
    inference_sync.start()

    logger.info("启动 FastAPI 服务器")
    import uvicorn
    # 端口从配置读取（config.yaml → app.port），默认 5002
    web_port = int(config_dict.get("app", {}).get("port", 5002))
    uvicorn.run(app, host="0.0.0.0", port=web_port, log_level="warning")

    inference_sync.stop()
    event_bus.stop()


def main():
    """入口主控制进程：只启动 Web 服务进程（0 MB 显存占用），等待用户在前端点击'开始测试'按需启动推理子进程."""
    config_path = args.config or os.path.join(BASE_DIR, "config.yaml")
    config = ConfigManager(config_path)
    paths = PathConfig.from_config(config.to_dict(), BASE_DIR)

    logger.info(f"FPS={config.fps}, GPU={config.gpu}")
    logger.info(f"数据目录: {paths.data_root}")
    logger.info(f"模型目录: {paths.model_root}")
    logger.info(f"结果目录: {paths.result_root}")

    cap = cv2.VideoCapture(config.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.fps
    cap.release()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = str(paths.get_result_dir(run_id))
    for sub in ("voice", "tracker", "gaze", "behavior", "qwen"):
        os.makedirs(os.path.join(result_dir, sub), exist_ok=True)
    logger.info(f"结果目录: {result_dir}")

    log_file = os.path.join(result_dir, "run.log")
    add_root_file_handler(log_file)
    logger.info(f"日志文件: {log_file}")

    config_dict = config.to_dict()
    config_dict["fps"] = fps
    config_dict["gpu"] = args.gpu
    config_dict["gpu_map"] = config.gpu_map
    config_dict["gpu_default"] = args.gpu or config.gpu_default
    # Redis 配置: 与模块级常量同源(config.yaml 的 redis 段 + HEDIAN_REDIS_* 环境变量),
    # 保证 Web 进程(清理/状态)与各推理子进程(EventBus/InferenceStream)指向同一实例与库
    config_dict["redis_host"] = REDIS_HOST
    config_dict["redis_port"] = REDIS_PORT
    config_dict["redis_db"] = REDIS_DB
    config_dict["_log_file"] = log_file
    config_dict["video_path"] = config.video_path
    paths_dict = {
        "data_root": str(paths.data_root),
        "model_root": str(paths.model_root),
        "result_root": str(paths.result_root),
        "base_dir": str(paths.base_dir),
    }

    # 启动前清理残留: 只删本项目命名空间, 不使用 flushdb(避免误删同实例其他业务的库)
    from core.redis_conn import clear_project_keys
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    clear_project_keys(redis_client)
    redis_client.close()
    logger.info(
        f"已清理 Redis({REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}) 残留 key，准备启动 Web 服务进程"
    )

    run_web_process(config_dict, paths_dict, run_id)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
