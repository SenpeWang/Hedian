"""
模块同步器 — 由 Web/主管理进程使用。从 Redis 读取各模块的进度和事件，进行时间对齐并推送。
"""
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Set

import redis

logger = logging.getLogger("core.module_sync")


class ModuleSync:
    """
    模块同步器 (时钟对齐 & 读取推送模式)
    """

    def __init__(
        self,
        fps: float = 30.0,
        immediate_types: Optional[Set[str]] = None,
        expected_modules: Optional[Set[str]] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        **kwargs
    ):
        self.fps = fps
        self.frame_interval = 1.0 / fps if fps > 0 else 1.0 / 30.0

        self._redis = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db,
            decode_responses=True, socket_connect_timeout=5,
        )
        self._redis.ping()
        logger.info("ModuleSync (模块同步器) Redis 连接成功")

        # Redis keys
        self._KEY_PROGRESS = "inference:progress"
        self._KEY_SNAPSHOT = "inference:snapshot"
        self._KEY_EVENT_STREAM = "inference:results:all"
        self._KEY_CLOCK = "inference:global_sec"

        # 主管理进程负责初始化时清理旧数据
        self._redis.delete(self._KEY_PROGRESS, self._KEY_SNAPSHOT,
                          self._KEY_EVENT_STREAM, self._KEY_CLOCK)

        self._immediate_types: Set[str] = immediate_types if immediate_types is not None else {
            "status", "progress", "video_start", "done", "report_progress",
        }

        self._push_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        self._expected_modules: Set[str] = expected_modules if expected_modules is not None else set()
        self._module_timeout: float = 60.0
        self._module_last_update: Dict[str, float] = {}

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._event_counter = 0

    def update_module_time(self, module_name: str, sec: float) -> None:
        """同步模式不执行此操作"""
        pass

    def update_module_snapshot(self, module_name: str, snapshot: Dict[str, Any]) -> None:
        """同步模式不执行此操作"""
        pass

    def remove_module(self, module_name: str) -> None:
        """从进度追踪中移除模块"""
        try:
            self._redis.hdel(self._KEY_PROGRESS, module_name)
            self._module_last_update.pop(module_name, None)
        except Exception as e:
            logger.error(f"从同步器中移除模块失败: {e}")

    def _compute_global_sec(self) -> float:
        """计算全局时钟：取所有未超时且预期的运行模块的最慢 local_sec 进度"""
        try:
            now = time.time()
            all_progress = self._redis.hgetall(self._KEY_PROGRESS)
            for name in all_progress.keys():
                if name not in self._module_last_update:
                    self._module_last_update[name] = now

            timed_out = []
            for name, last_update in list(self._module_last_update.items()):
                if now - last_update > self._module_timeout:
                    timed_out.append(name)
            for name in timed_out:
                logger.warning(f"模块 {name} 超时，自动从进度跟踪中移除")
                self._redis.hdel(self._KEY_PROGRESS, name)
                self._module_last_update.pop(name, None)

            if not all_progress:
                return float("inf")

            if self._expected_modules:
                relevant = {k: float(v) for k, v in all_progress.items() if k in self._expected_modules}
            else:
                relevant = {k: float(v) for k, v in all_progress.items()}

            if not relevant:
                return float("inf")

            if self._expected_modules:
                registered = set(all_progress.keys())
                if not self._expected_modules.issubset(registered):
                    if not registered:
                        return 0.0
                    relevant = {k: float(v) for k, v in all_progress.items() if k in registered}
                    if not relevant:
                        return 0.0

            return min(relevant.values())
        except Exception as e:
            logger.error(f"计算全局时钟失败: {e}")
            return 0.0

    def _get_context(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模块的快照"""
        try:
            snapshots = self._redis.hgetall(self._KEY_SNAPSHOT)
            result = {}
            for k, v in snapshots.items():
                try:
                    result[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    result[k] = {}
            return result
        except Exception as e:
            logger.error(f"获取上下文快照失败: {e}")
            return {}

    def set_push_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """设置推送回调函数"""
        self._push_callback = callback

    def start(self) -> None:
        """启动事件对齐聚合线程"""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True

        self._thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self._thread.start()
        logger.info("ModuleSync 线程启动")

    def stop(self) -> None:
        """停止事件对齐聚合线程"""
        if not self._running:
            return
        self._stop_event.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._flush_remaining_events()
        logger.info("ModuleSync 线程已停止")

    def push_sentinel(self) -> None:
        """推送终止信号"""
        if self._push_callback is not None:
            try:
                self._push_callback(None)
            except Exception:
                pass

    def _aggregation_loop(self) -> None:
        """对齐推送循环"""
        while not self._stop_event.is_set():
            loop_start = time.time()
            try:
                # 流水线结束检测：main 进程在 join 完所有模块进程后，
                # 会设置 pipeline:status=done。此时各模块已写完全部事件，
                # 直接 flush 剩余事件并推送终止信号，让前端恢复交互。
                if self._redis.get("pipeline:status") == "done":
                    logger.info("检测到 pipeline:status=done，刷新剩余事件并推送终止信号")
                    self._flush_remaining_events()
                    self.push_sentinel()
                    break

                global_sec = self._compute_global_sec()
                if global_sec != float("inf"):
                    logger.debug(f"计算全局时钟: global_sec={global_sec:.2f}")
                    self._push_events_up_to(global_sec)
                    self._redis.set(self._KEY_CLOCK, str(global_sec), ex=10)
                    self._do_push({"type": "clock_sync", "localSec": global_sec})
            except Exception as e:
                logger.error(f"聚合循环中发生异常: {e}")

            elapsed = time.time() - loop_start
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _push_events_up_to(self, global_sec: float) -> None:
        """获取并推送所有 local_sec <= global_sec 的事件"""
        events_to_push = []
        ids_to_delete = []

        try:
            entries = self._redis.xrange(self._KEY_EVENT_STREAM, min="-", max="+")
            all_events = []
            for entry_id, fields in entries:
                try:
                    ev = json.loads(fields["payload"])
                    local_sec = float(fields.get("local_sec", 0))
                    all_events.append((local_sec, ev, entry_id))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"解析事件失败: {e}")
                    ids_to_delete.append(entry_id)

            all_events.sort(key=lambda x: x[0])

            for local_sec, ev, entry_id in all_events:
                if local_sec <= global_sec:
                    if ev.get("type") == "video_frame":
                        frame_key = f"inference:video_frame:{local_sec}"
                        frame_data = self._redis.get(frame_key)
                        if frame_data:
                            ev["frame_data"] = frame_data
                            self._redis.delete(frame_key)
                    events_to_push.append(ev)
                    ids_to_delete.append(entry_id)
                else:
                    break

            if ids_to_delete:
                for entry_id in ids_to_delete:
                    self._redis.xdel(self._KEY_EVENT_STREAM, entry_id)

            if events_to_push:
                logger.info(f"推送对齐事件: {len(events_to_push)} 个, global_sec={global_sec:.2f}")
                context = self._get_context()
                for event in events_to_push:
                    event["context"] = context
                    self._do_push(event)

        except Exception as e:
            logger.error(f"对齐并推送事件失败: {e}")

    def _flush_remaining_events(self) -> None:
        """强制刷新剩余的所有事件"""
        try:
            entries = self._redis.xrange(self._KEY_EVENT_STREAM, min="-", max="+")
            context = self._get_context()
            for entry_id, fields in entries:
                try:
                    ev = json.loads(fields["payload"])
                    ev["context"] = context
                    self._do_push(ev)
                except (json.JSONDecodeError, TypeError):
                    pass
                self._redis.xdel(self._KEY_EVENT_STREAM, entry_id)
        except Exception as e:
            logger.error(f"清理剩余事件失败: {e}")

    def _do_push(self, event: Dict[str, Any]) -> None:
        """执行实际的回调推送"""
        if self._push_callback is not None:
            try:
                self._push_callback(event)
            except Exception as e:
                logger.error(f"调用推送回调失败: {e}")

    def push_display(self, event_type: str, data: Dict[str, Any]) -> None:
        """同步器不需要推送结果"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """获取统计状态"""
        try:
            all_progress = self._redis.hgetall(self._KEY_PROGRESS)
            all_snapshots = self._redis.hgetall(self._KEY_SNAPSHOT)
            stream_len = self._redis.xlen(self._KEY_EVENT_STREAM)
            return {
                "stream_size": stream_len,
                "global_sec": self._compute_global_sec(),
                "module_times": {k: float(v) for k, v in all_progress.items()},
                "module_snapshots": {k: json.loads(v) for k, v in all_snapshots.items()},
                "running": self._running,
                "fps": self.fps,
            }
        except Exception as e:
            logger.error(f"获取同步器状态失败: {e}")
            return {}

    def clear(self) -> None:
        """清空数据"""
        try:
            self._redis.delete(self._KEY_PROGRESS, self._KEY_SNAPSHOT,
                              self._KEY_EVENT_STREAM, self._KEY_CLOCK)
            self._event_counter = 0
        except Exception as e:
            logger.error(f"清空 Redis 状态失败: {e}")
