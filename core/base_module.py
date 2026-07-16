"""
模块基类 — 统一所有业务模块的接口

所有业务模块（Voice, MOT, Gaze, Behavior）继承此基类，
实现统一的初始化、处理、保存接口。
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging
import time

from core.event_bus import EventBus
from core.display_buffer import DisplayBuffer
from core.path_manager import PathConfig


class BaseModule(ABC):
    """
    业务模块基类

    所有业务模块必须继承此类并实现以下方法：
    - module_name: 模块名称
    - initialize(): 初始化模块
    - process_video(): 处理视频
    - save_results(): 保存结果

    使用方式：
        module = MyModule(event_bus, config, paths, display_buffer)
        module.start(video_path, run_id)
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: dict,
        paths: PathConfig,
        display_buffer: DisplayBuffer,
    ):
        """
        初始化模块

        Args:
            event_bus: 消息总线
            config: 配置字典
            paths: 路径配置
            display_buffer: 前端同步器
        """
        self.event_bus = event_bus
        self.config = config
        self.paths = paths
        self.display_buffer = display_buffer
        self.logger = logging.getLogger(f"module.{self.module_name}")
        self._running = False
        self._start_time = 0.0

    @property
    @abstractmethod
    def module_name(self) -> str:
        """
        模块名称

        Returns:
            模块名称，如 'voice', 'tracker', 'gaze', 'behavior'
        """
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化模块（加载模型等）

        Returns:
            初始化是否成功
        """
        pass

    @abstractmethod
    def process_video(self, video_path: str) -> None:
        """
        处理视频

        Args:
            video_path: 视频文件路径
        """
        pass

    @abstractmethod
    def save_results(self, run_id: str) -> None:
        """
        保存结果

        Args:
            run_id: 运行 ID
        """
        pass

    def start(self, video_path: str, run_id: str) -> None:
        """
        启动模块（模板方法）

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

            # 2. 注册到聚合器
            self.display_buffer.update_module_time(self.module_name, 0.0)

            # 3. 处理视频
            self.logger.info(f"处理视频: {video_path}")
            self.process_video(video_path)

            # 4. 保存结果
            self.logger.info("保存结果...")
            self.save_results(run_id)

            elapsed = time.time() - self._start_time
            self.logger.info(f"模块 {self.module_name} 完成，耗时 {elapsed:.1f}s")

        except Exception as e:
            self.logger.error(f"模块 {self.module_name} 错误: {e}", exc_info=True)
        finally:
            self._running = False
            # 不移除模块，保留进度参与全局对齐
            # 模块完成后，其进度保持在最终值
            # global_sec 会继续考虑该模块的进度
            # 直到所有模块都完成，事件才会全部推送

    def stop(self) -> None:
        """停止模块"""
        self._running = False
        self.logger.info(f"模块 {self.module_name} 停止")

    @property
    def is_running(self) -> bool:
        """模块是否正在运行"""
        return self._running

    def update_progress(self, current: float, total: float = None) -> None:
        """
        更新处理进度

        Args:
            current: 当前处理到的时间点（秒）
            total: 总时长（秒），用于计算百分比
        """
        self.display_buffer.update_module_time(self.module_name, current)
        # 推送进度事件到前端（每秒最多一次）
        if total and total > 0:
            now = time.time()
            if not hasattr(self, '_last_progress_push'):
                self._last_progress_push = 0
            if now - self._last_progress_push >= 1.0:
                self._last_progress_push = now
                pct = min(100, int(current / total * 100))
                self.push_display("progress", {
                    "label": self.module_name,
                    "pct": pct,
                    "current": round(current, 1),
                    "total": round(total, 1),
                })

    def push_display(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        推送数据到推理流（用于后端时间对齐，前端展示）

        Args:
            event_type: 推理事件类型，如 'tracking', 'gaze', 'voice', 'behavior'
            data: 数据内容
        """
        self.display_buffer.push_display(event_type, data)

    def push_event(self, msg_type: str, data: Dict[str, Any], ts: float = 0.0) -> None:
        """
        推送指令到跨进程消息流（用于模块间通信，触发模块调用）

        Args:
            msg_type: 消息类型，定义在 EventTopic 中
            data: 消息数据内容
            ts: 时间戳
        """
        self.event_bus.publish(msg_type, data, ts=ts)

