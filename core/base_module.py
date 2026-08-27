"""
模块基类 — 统一所有业务模块的接口.

所有业务模块（Voice, MOT, Gaze, Behavior）继承此基类，
实现统一的初始化、处理、保存接口。
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import logging
import time

from core.event_bus import EventStream, EventTopic
from core.inference_stream import InferenceStream
from core.inference_sync import InferenceSync
from core.path_manager import PathManager


class BaseModule(ABC):
    """
    业务模块基类.

    所有业务模块必须继承此类并实现以下方法：
    - module_name: 模块名称
    - initialize(): 初始化模块
    - process_video(): 处理视频
    - save_results(): 保存结果

    使用方式：
        module = MyModule(event_bus, config, paths, inference_stream)
        module.start(video_path, run_id)
    """

    def __init__(
        self,
        event_bus: EventStream,
        config: dict,
        paths: PathManager,
        inference_stream: Union[InferenceStream, InferenceSync],
    ):
        """
        初始化模块.

        Args:
            event_bus: 消息总线
            config: 配置字典
            paths: 路径配置
            inference_stream: 推理流写入端（InferenceStream，模块进程）或同步器（InferenceSync，Web 进程）
        """
        self.event_bus = event_bus
        self.config = config
        self.paths = paths
        self.inference_stream = inference_stream
        self.logger = logging.getLogger(f"module.{self.module_name}")
        self._running = False
        self._start_time = 0.0
        self._run_id = None
        # 本模块产出的 source 集合（push_display 自动登记）：用于退出时上报结束信号
        self._inference_sources: set = set()
        # 仅代推、不归属本模块的 source：退出时不标记结束（其生命周期归所属模块）
        self._borrowed_sources: set = set()
        # 归属本模块但进度独立写入的 source：update_progress 时跳过（如 gaze 由内嵌组件异步独立写进度）
        self._independent_progress_sources: set = set()
        # 进度推送与对齐日志的节流时间戳（统一在 __init__ 初始化，避免方法内 hasattr 懒初始化）
        self._last_progress_push = 0.0
        self._last_align_log_ts = 0.0

        # 订阅评估器触发的即时保存事件
        self.event_bus.subscribe(
            EventTopic.SAVE_KEY_MOMENTS, self._on_save_key_moments
        )

    def _on_save_key_moments(self, msg: dict) -> None:
        """
        响应评估器的 SAVE_KEY_MOMENTS 事件：立即保存当前 key_moments.

        这样在 FLOW_ENDED 触发时，各模块的数据已经落盘，评估器可以马上读取。
        """
        if self._run_id:
            try:
                self.save_results(self._run_id)
                self.logger.info(f"响应 SAVE_KEY_MOMENTS，已保存 {self.module_name} 结果")
            except Exception as e:
                self.logger.error(f"响应 SAVE_KEY_MOMENTS 保存失败: {e}", exc_info=True)

    @property
    @abstractmethod
    def module_name(self) -> str:
        """
        模块名称.

        Returns:
            模块名称，如 'voice', 'tracker', 'gaze', 'behavior'
        """
        pass

    @property
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化模块（加载模型等.

        Returns:
            初始化是否成功
        """
        pass

    @abstractmethod
    def process_video(self, video_path: str) -> None:
        """
        处理视频.

        Args:
            video_path: 视频文件路径
        """
        pass

    @abstractmethod
    def save_results(self, run_id: str) -> None:
        """
        保存结果.

        Args:
            run_id: 运行 ID
        """
        pass

    def start(self, video_path: str, run_id: str) -> None:
        """
        启动模块（模板方法.

        按顺序执行：初始化 → 注册到聚合器 → 处理视频 → 保存结果

        Args:
            video_path: 视频文件路径
            run_id: 运行 ID
        """
        self.logger.info(f"模块 {self.module_name} 启动")
        self._running = True
        self._start_time = time.time()
        self._run_id = run_id  # 存储 run_id 供 process_video 使用

        try:
            # 1. 初始化
            self.logger.info("初始化中...")
            if not self.initialize():
                self.logger.error("初始化失败")
                return

            # 2. 处理视频（过程中通过 push_display 自动登记 source，并 per-source 写进度）
            self.logger.info(f"处理视频: {video_path}")
            self.process_video(video_path)

            # 3. 保存结果
            self.logger.info("保存结果...")
            self.save_results(run_id)

            elapsed = time.time() - self._start_time
            self.logger.info(f"模块 {self.module_name} 完成，耗时 {elapsed:.1f}s")

        except Exception as e:
            self.logger.error(f"模块 {self.module_name} 错误: {e}", exc_info=True)
        finally:
            self._running = False
            # 退出即上报所有归属 source 的结束信号（借用 source 排除，由所属模块负责）
            try:
                if hasattr(self.inference_stream, 'mark_owned_done'):
                    owned = self._inference_sources - self._borrowed_sources
                    self.inference_stream.mark_owned_done(owned)
            except Exception as e:
                self.logger.warning(f"模块 {self.module_name} 上报 source 结束信号失败: {e}")

    def stop(self) -> None:
        """停止模块."""
        self._running = False
        self.logger.info(f"模块 {self.module_name} 停止")

    @property
    def is_running(self) -> bool:
        """模块是否正在运行."""
        return self._running

    def update_progress(self, current: float, total: Optional[float] = None) -> None:
        """更新进度：per-source 写入（借用 source 与独立进度 source 跳过."""
        for source in self._inference_sources:
            if source in self._borrowed_sources or source in self._independent_progress_sources:
                continue
            self.inference_stream.update_module_time(source, current)
        # 推送进度事件到前端（每 0.3 秒高频更新）
        if total and total > 0:
            now = time.time()
            if now - self._last_progress_push >= 0.3:
                self._last_progress_push = now
                pct = min(100.0, current / total * 100.0)
                self.push_display("progress", {
                    "localSec": round(current, 2),
                    "tag": "progress",
                    "data": {"label": self.module_name, "pct": round(pct, 1)},
                })

    def push_display(self, event_type: str, data: Dict[str, Any]) -> None:
        """推送数据到推理流（非即时类型自动登记为归属 source."""
        if event_type not in ("progress", "video_start"):
            self._inference_sources.add(event_type)
        self.inference_stream.push_display(event_type, data)

    def push_event(self, msg_type: str, data: Dict[str, Any], ts: float = 0.0) -> None:
        """推送指令到跨进程消息流."""
        self.event_bus.publish(msg_type, data, ts=ts)

