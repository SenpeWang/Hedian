"""
事件聚合器 — 基于 Redis 的跨进程时间对齐

核心原则：后端各模块异步推理，各自产生带 local_sec 的局部事件，
聚合器按全局时钟 global_sec 对齐后才推送到前端。

全局时钟策略：
  global_sec 取所有已注册模块中最慢的那个（min），
  确保只有所有模块都处理到某个时间点后，该时间点之前的事件才会被推送。

跨进程通信：
  - 模块进度通过 Redis Hash 存储
  - 事件通过 Redis Stream 存储
  - 聚合器轮询 Redis 判断是否可以推送
"""
import json
import threading
import time
import logging
from typing import Callable, Dict, Any, Optional, Set

import redis

logger = logging.getLogger("core.frontend_sync")


class FrontendSync:
    """
    事件聚合器（Redis 版）

    负责将多个模块的事件按时间对齐后推送到前端。
    """

    def __init__(
        self,
        fps: float = 30.0,
        immediate_types: Optional[Set[str]] = None,
        expected_modules: Optional[Set[str]] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        writer_only: bool = False,
    ):
        self.fps = fps
        self.frame_interval = 1.0 / fps if fps > 0 else 1.0 / 30.0

        # Redis 连接
        self._redis = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db,
            decode_responses=True, socket_connect_timeout=5,
        )
        self._redis.ping()

        # writer_only 模式: 只写入 Redis Stream，不运行聚合循环
        self._writer_only = writer_only
        if writer_only:
            logger.info(f"FrontendSync Redis 连接成功 (writer_only 模式)")
        else:
            logger.info(f"FrontendSync Redis 连接成功")

        # Redis key 前缀
        self._KEY_PROGRESS = "aggregator:progress"       # Hash: 模块进度
        self._KEY_SNAPSHOT = "aggregator:snapshot"       # Hash: 模块快照
        self._KEY_EVENT_STREAM = "aggregator:events"     # Stream: 事件流
        self._KEY_CLOCK = "aggregator:global_sec"        # String: 全局时钟

        # 清理旧数据
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

        # 事件计数器
        self._event_counter = 0

    def update_module_time(self, module_name: str, sec: float) -> None:
        """
        更新某个模块的当前处理进度（local_sec）

        Args:
            module_name: 模块名称
            sec: 该模块当前处理到的时间点（秒）
        """
        self._redis.hset(self._KEY_PROGRESS, module_name, str(sec))
        self._module_last_update[module_name] = time.time()

    def update_module_snapshot(self, module_name: str, snapshot: Dict[str, Any]) -> None:
        """
        更新某个模块的最新状态快照

        Args:
            module_name: 模块名称
            snapshot: 该模块的最新状态
        """
        self._redis.hset(self._KEY_SNAPSHOT, module_name,
                         json.dumps(snapshot, ensure_ascii=False))

    def remove_module(self, module_name: str) -> None:
        """
        移除一个模块的进度追踪

        Args:
            module_name: 模块名称
        """
        self._redis.hdel(self._KEY_PROGRESS, module_name)
        self._module_last_update.pop(module_name, None)

    def _compute_global_sec(self) -> float:
        """
        计算全局时钟：取所有已注册模块中最慢的那个

        Returns:
            全局时钟（秒）
        """
        # 检查模块超时
        now = time.time()
        timed_out = []
        for name, last_update in self._module_last_update.items():
            if now - last_update > self._module_timeout:
                timed_out.append(name)
        for name in timed_out:
            logger.warning(f"模块 {name} 超时，自动移除")
            self._redis.hdel(self._KEY_PROGRESS, name)
            self._module_last_update.pop(name, None)

        # 获取所有模块进度
        all_progress = self._redis.hgetall(self._KEY_PROGRESS)
        if not all_progress:
            return float("inf")

        # 只用预期模块计算
        if self._expected_modules:
            relevant = {k: float(v) for k, v in all_progress.items()
                       if k in self._expected_modules}
        else:
            relevant = {k: float(v) for k, v in all_progress.items()}

        if not relevant:
            return float("inf")

        # 如果有预期模块，检查是否全部注册
        if self._expected_modules:
            registered = set(all_progress.keys())
            logger.info(f"预期模块: {self._expected_modules}, 已注册: {registered}, 子集检查: {self._expected_modules.issubset(registered)}")
            # 如果有预期模块但没有全部注册，使用已注册的模块计算
            # （模块完成后会被移除，但其事件已经处理完毕）
            if not self._expected_modules.issubset(registered):
                # 如果没有任何模块注册，返回无穷大（所有事件都可以推送）
                if not registered:
                    return float("inf")
                # 如果有部分模块注册，使用已注册的模块计算
                relevant = {k: float(v) for k, v in all_progress.items() if k in registered}
                if not relevant:
                    return float("inf")

        result = min(relevant.values())
        logger.info(f"global_sec计算结果: {result:.2f}")
        return result

    def _get_context(self) -> Dict[str, Dict[str, Any]]:
        """获取当前所有模块的最新快照"""
        snapshots = self._redis.hgetall(self._KEY_SNAPSHOT)
        result = {}
        for k, v in snapshots.items():
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                result[k] = {}
        return result

    def set_push_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """设置推送回调函数"""
        self._push_callback = callback

    def push_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        接收模块事件

        Args:
            event_type: 事件类型
            data: 事件数据
        """
        ev = {"type": event_type, **data}
        logger.info(f"收到事件: type={event_type}, localSec={data.get('localSec', 'N/A')}")

        if event_type in self._immediate_types:
            if "localSec" in ev:
                ev["context"] = self._get_context()
            # writer_only 模式下，立即事件也写入 Stream（由 reader 统一推送）
            if self._writer_only:
                self._event_counter += 1
                local_sec = ev.get("localSec", 0)
                self._redis.xadd(self._KEY_EVENT_STREAM, {
                    "local_sec": str(local_sec),
                    "counter": str(self._event_counter),
                    "payload": json.dumps(ev, ensure_ascii=False),
                })
            else:
                self._do_push(ev)
            return

        if "localSec" not in ev:
            logger.error(f"时间对齐事件类型 '{event_type}' 必须包含 'localSec' 字段")
            return

        # 写入 Redis Stream
        self._event_counter += 1
        local_sec = ev["localSec"]
        self._redis.xadd(self._KEY_EVENT_STREAM, {
            "local_sec": str(local_sec),
            "counter": str(self._event_counter),
            "payload": json.dumps(ev, ensure_ascii=False),
        })

    def push_sentinel(self) -> None:
        """推送流终止信号"""
        if self._push_callback is not None:
            try:
                self._push_callback(None)
            except Exception:
                pass

    def start(self) -> None:
        """启动聚合器线程"""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True

        if self._writer_only:
            # writer_only 模式: 不启动聚合循环，只写入 Redis Stream
            logger.info("FrontendSync 启动 (writer_only 模式，不运行聚合循环)")
            return

        self._thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self._thread.start()
        logger.info("事件聚合器启动")

    def stop(self) -> None:
        """停止聚合器线程"""
        if not self._running:
            return
        self._stop_event.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._flush_remaining_events()
        logger.info("事件聚合器停止")

    def _aggregation_loop(self) -> None:
        """聚合主循环"""
        while not self._stop_event.is_set():
            loop_start = time.time()

            global_sec = self._compute_global_sec()
            logger.info(f"global_sec={global_sec:.2f}")
            self._push_events_up_to(global_sec)

            # 更新全局时钟到 Redis
            self._redis.set(self._KEY_CLOCK, str(global_sec), ex=10)

            # 推送时钟同步事件
            self._do_push({"type": "clock_sync", "localSec": global_sec})

            elapsed = time.time() - loop_start
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _push_events_up_to(self, global_sec: float) -> None:
        """推送所有事件（不做时间过滤，前端按 localSec 排序）"""
        events_to_push = []
        ids_to_delete = []

        # 从 Redis Stream 读取所有事件
        entries = self._redis.xrange(self._KEY_EVENT_STREAM, min="-", max="+")

        # 收集所有事件
        for entry_id, fields in entries:
            try:
                ev = json.loads(fields["payload"])
                events_to_push.append(ev)
                ids_to_delete.append(entry_id)
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"解析事件失败: {e}")
                ids_to_delete.append(entry_id)

        # 批量删除已处理的事件
        if ids_to_delete:
            for entry_id in ids_to_delete:
                self._redis.xdel(self._KEY_EVENT_STREAM, entry_id)

        # 推送所有事件
        if events_to_push:
            logger.info(f"推送 {len(events_to_push)} 个事件")
            context = self._get_context()
            for event in events_to_push:
                event["context"] = context
                self._do_push(event)

    def _flush_remaining_events(self) -> None:
        """刷新剩余事件"""
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

    def _do_push(self, event: Dict[str, Any]) -> None:
        """执行推送"""
        if self._push_callback is not None:
            try:
                self._push_callback(event)
                if event.get("type") not in ("clock_sync", "progress"):
                    logger.info(f"推送到前端: type={event.get('type')}, localSec={event.get('localSec', 'N/A')}")
            except Exception as e:
                logger.error(f"推送事件失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取聚合器统计"""
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

    def clear(self) -> None:
        """清空所有事件和模块信息"""
        self._redis.delete(self._KEY_PROGRESS, self._KEY_SNAPSHOT,
                          self._KEY_EVENT_STREAM, self._KEY_CLOCK)
        self._event_counter = 0
