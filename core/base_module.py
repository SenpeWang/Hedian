"""
模块基类 — 统一所有业务模块的接口

所有业务模块（Voice, MOT, Gaze, Behavior）继承此基类，
实现统一的初始化、处理、保存接口。
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import logging
import time
import redis

from core.event_bus import EventBus, EventTopic
from core.inference_bus import InferenceBus
from core.module_sync import ModuleSync
from core.path_manager import PathConfig


# 全局对齐参数：任何模块推理进度不能比最慢的模块快超过此秒数
ALIGN_MAX_LEAD_SEC = 60.0
# 单次对齐等待最长秒数
ALIGN_WAIT_TIMEOUT_SEC = 30.0
_ALIGN_REDIS_HOST = "localhost"
_ALIGN_REDIS_PORT = 6379
_ALIGN_REDIS_DB = 0
_ALIGN_PROGRESS_KEY = "inference:progress"


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
        display_buffer: Union[InferenceBus, ModuleSync],
    ):
        """
        初始化模块

        Args:
            event_bus: 消息总线
            config: 配置字典
            paths: 路径配置
            display_buffer: 推理总线（写模式）或模块同步器（读模式）
        """
        self.event_bus = event_bus
        self.config = config
        self.paths = paths
        self.display_buffer = display_buffer
        self.logger = logging.getLogger(f"module.{self.module_name}")
        self._running = False
        self._start_time = 0.0
        self._run_id = None

        # 订阅评估器触发的即时保存事件
        self.event_bus.subscribe(
            EventTopic.SAVE_KEY_MOMENTS, self._on_save_key_moments
        )

    def _on_save_key_moments(self, msg: dict) -> None:
        """
        响应评估器的 SAVE_KEY_MOMENTS 事件：立即保存当前 key_moments。

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
        模块名称

        Returns:
            模块名称，如 'voice', 'tracker', 'gaze', 'behavior'
        """
        pass

    @property
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
            # 保留模块进度参与全局对齐；ModuleSync 会基于时间戳自动剔除长期未推进的模块

    def stop(self) -> None:
        """停止模块"""
        self._running = False
        self.logger.info(f"模块 {self.module_name} 停止")

    @property
    def is_running(self) -> bool:
        """模块是否正在运行"""
        return self._running

    def update_progress(self, current: float, total: float = None) -> None:
        """更新处理进度，内置限速对齐"""
        self.display_buffer.update_module_time(self.module_name, current)
        # 推送进度事件到前端（每秒最多一次）
        if total and total > 0:
            now = time.time()
            if not hasattr(self, '_last_progress_push'):
                self._last_progress_push = 0
            if now - self._last_progress_push >= 1.0:
                self._last_progress_push = now
                pct = min(100, current / total * 100)
                self.push_display("progress", {
                    "localSec": round(current, 2),
                    "tag": "progress",
                    "data": {"label": self.module_name, "pct": round(pct, 1)},
                })
                # 通用对齐限速
                self.align_with_slowest_module(current)

    def push_display(self, event_type: str, data: Dict[str, Any]) -> None:
        """推送数据到推理流"""
        self.display_buffer.push_display(event_type, data)

    def push_event(self, msg_type: str, data: Dict[str, Any], ts: float = 0.0) -> None:
        """推送指令到跨进程消息流"""
        self.event_bus.publish(msg_type, data, ts=ts)

    def align_with_slowest_module(self, my_progress_sec: float) -> None:
        """通用对齐限速：本模块不能比最慢模块快超过 ALIGN_MAX_LEAD_SEC 秒"""
        try:
            from core.redis_conn import get_redis_client
            r = get_redis_client(host=_ALIGN_REDIS_HOST, port=_ALIGN_REDIS_PORT, db=_ALIGN_REDIS_DB)
            wait_start_ts = time.time()
            while True:
                progress_map = r.hgetall(_ALIGN_PROGRESS_KEY)
                # 收集除自身外其他模块的进度
                other_progress_secs = []
                for name, val in progress_map.items():
                    if name == self.module_name:
                        continue
                    try:
                        other_progress_secs.append(float(val))
                    except ValueError:
                        pass

                if not other_progress_secs:
                    # 其他模块尚未启动，短暂等待
                    if time.time() - wait_start_ts > ALIGN_WAIT_TIMEOUT_SEC:
                        self.logger.warning(
                            f"模块 {self.module_name} 等待其他模块启动超时(>{ALIGN_WAIT_TIMEOUT_SEC}s)，继续推理"
                        )
                        break
                    time.sleep(0.1)
                    continue

                slowest_other_sec = min(other_progress_secs)
                lead_sec = my_progress_sec - slowest_other_sec
                if lead_sec <= ALIGN_MAX_LEAD_SEC:
                    break
                # 超时保护：防止最慢模块异常卡死时本模块永久阻塞
                if time.time() - wait_start_ts > ALIGN_WAIT_TIMEOUT_SEC:
                    self.logger.warning(
                        f"模块 {self.module_name} 对齐限速超时(>{ALIGN_WAIT_TIMEOUT_SEC}s)："
                        f"self={my_progress_sec:.1f}s, slowest_other={slowest_other_sec:.1f}s, "
                        f"领先{lead_sec:.1f}s > {ALIGN_MAX_LEAD_SEC}s，继续推理"
                    )
                    break
                if not hasattr(self, '_last_align_log_ts'):
                    self._last_align_log_ts = 0.0
                if time.time() - self._last_align_log_ts > 5.0:
                    self._last_align_log_ts = time.time()
                    self.logger.info(
                        f"模块 {self.module_name} 对齐限速：self={my_progress_sec:.1f}s, "
                        f"slowest_other={slowest_other_sec:.1f}s, 领先{lead_sec:.1f}s > {ALIGN_MAX_LEAD_SEC}s，等待"
                    )
                time.sleep(0.1)
            r.close()
        except Exception as e:
            self.logger.warning(f"模块 {self.module_name} 对齐限速异常（不阻塞推理）: {e}")

