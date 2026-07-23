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

# 全局对齐容忍窗口（秒）
# 设计意图：允许 global_sec 比最慢模块超前最多此秒数，避免慢模块短暂卡顿时
# 前端被冻结、快模块的事件无法推送。配合 base_module 的 ALIGN_MAX_LEAD_SEC=60s
# 限速器，既能防止 Redis Stream 无限堆积，又能让前端不会因慢模块卡死。
# 工业界滑动窗口对齐的标准做法：允许模态在窗口内缺失，不阻塞全局。
GLOBAL_ALIGN_TOLERANCE_SEC = 5.0


class ModuleSync:
    """模块同步器"""

    def __init__(
        self,
        fps: float = 30.0,
        immediate_types: Optional[Set[str]] = None,
        expected_modules: Optional[Set[str]] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        duration: float = 0.0,
        **kwargs
    ):
        self.fps = fps
        self.duration = duration
        self.frame_interval = 1.0 / fps if fps > 0 else 1.0 / 30.0

        from core.redis_conn import get_redis_client
        self._redis = get_redis_client(host=redis_host, port=redis_port, db=redis_db)
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
            "progress", "video_start",
        }

        self._push_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        self._expected_modules: Set[str] = expected_modules if expected_modules is not None else set()
        self._module_timeout: float = 60.0
        self._module_last_update: Dict[str, float] = {}
        # 跟踪上次观察到的进度值，仅在值变化时更新时间戳，避免死模块被误判为活跃
        self._module_last_value: Dict[str, str] = {}

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._event_counter = 0
        # 增量读取 Stream 的上次位置（闭区间，下次读取时跳过该 id）
        self._last_stream_id: str = "0-0"
        # 本地暂存 local_sec > 当前推送时间的事件，下次循环再处理
        self._pending_events: list = []
        # 已推送到的 global_sec，按帧率步长推进，避免突发推送
        self._pushed_global_sec: float = 0.0

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
            self._module_last_value.pop(module_name, None)
        except Exception as e:
            logger.error(f"从同步器中移除模块失败: {e}")

    def _compute_global_sec(self) -> float:
        """计算全局时钟：取所有未超时且预期的运行模块的最慢 local_sec 进度

        时间戳判定：仅当进度值实际变化时才更新 last_update，否则保持原值，
        这样真正停止推进的模块会在 _module_timeout 后被剔除，避免死模块污染全局时钟。
        """
        try:
            now = time.time()
            all_progress = self._redis.hgetall(self._KEY_PROGRESS)

            # 仅在进度值变化时更新时间戳
            for name, value in all_progress.items():
                if name not in self._module_last_value or self._module_last_value[name] != value:
                    self._module_last_value[name] = value
                    self._module_last_update[name] = now

            # 已不在 Redis 中的模块清理本地状态
            for name in list(self._module_last_update.keys()):
                if name not in all_progress:
                    self._module_last_update.pop(name, None)
                    self._module_last_value.pop(name, None)

            timed_out = []
            for name, last_update in list(self._module_last_update.items()):
                try:
                    val = float(all_progress.get(name, 0.0))
                except ValueError:
                    val = 0.0
                
                # 如果该模块已经推进到了视频末尾（允许 1.5 秒误差），说明它已经正常结束，绝对不应该判定为超时
                if self.duration > 0 and val >= self.duration - 1.5:
                    continue

                if now - last_update > self._module_timeout:
                    timed_out.append(name)
            for name in timed_out:
                logger.warning(f"模块 {name} 超时（{self._module_timeout:.0f}s 无进度推进），自动从进度跟踪中移除")
                self._redis.hdel(self._KEY_PROGRESS, name)
                self._module_last_update.pop(name, None)
                self._module_last_value.pop(name, None)

            if not all_progress:
                return float("inf")

            if self._expected_modules:
                relevant = {k: float(v) for k, v in all_progress.items() if k in self._expected_modules}
            else:
                relevant = {k: float(v) for k, v in all_progress.items()}

            if not relevant:
                return float("inf")

            # 排除进度为0的模块：启动阶段某个模块（如voice首段转录）尚未产出进度时，
            # 不应拖累全局时钟导致前端冻结在0秒
            positive_progress = {k: v for k, v in relevant.items() if v > 0}
            if positive_progress:
                slowest_sec = min(positive_progress.values())
                fastest_sec = max(positive_progress.values())
                # 容忍窗口：允许 global_sec 比最慢模块超前最多 GLOBAL_ALIGN_TOLERANCE_SEC 秒
                # 但不能超过最快模块的进度（否则没有事件可推）
                # 这样慢模块短暂卡顿时，快模块在容忍窗口内的事件仍能推送给前端
                return min(slowest_sec + GLOBAL_ALIGN_TOLERANCE_SEC, fastest_sec)
            # 所有模块进度都为0时返回0，让早期事件正常推送
            return min(relevant.values())
        except Exception as e:
            logger.error(f"计算全局时钟失败: {e}")
            return float("inf")

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
        """对齐推送循环，推理结束后保持常驻等待下一次触发"""
        last_global_sec = -1.0
        stall_count = 0
        STALL_TIMEOUT = 30.0  # global_sec 停滞超过30秒视为本轮推理完成
        cycle_done = False    # 本轮推理是否已推送 done 哨兵

        while not self._stop_event.is_set():
            loop_start = time.time()
            try:
                global_sec = self._compute_global_sec()
                if global_sec != float("inf"):
                    # 检测 global_sec 是否停滞
                    if abs(global_sec - last_global_sec) < 0.01:
                        stall_count += 1
                    else:
                        stall_count = 0
                        last_global_sec = global_sec
                        cycle_done = False  # 进度更新，重置本轮完成标志

                    # 停滞超过30秒 → 检查前端是否已经追赶上 global_sec
                    if stall_count >= int(STALL_TIMEOUT / self.frame_interval) and not cycle_done:
                        # 关键保护：前端播放进度尚未追赶上全局时钟时绝不发送 done，否则剩余内容全部丢失。
                        # 注意：此处不做任何加速追赶（紧循环会跳过下方 sleep，导致最后一段内容
                        # 以数倍速闪播）——直接落到下方正常帧率推送，按 1 倍速平滑播完剩余内容，
                        # 待前端真正追上后由本分支发送 done。
                        if self._pushed_global_sec >= global_sec - 1.0:
                            logger.info(f"global_sec 停滞 {STALL_TIMEOUT}s 且前端已追赶完毕，本轮推理完成，刷新剩余事件并推送 done")
                            self._flush_remaining_events()
                            self.push_sentinel()
                            cycle_done = True
                            # 重置状态，准备等待下一次 /start 触发的新推理
                            last_global_sec = -1.0
                            stall_count = 0
                            # 重置推送限速状态，避免下一轮被旧值卡住
                            self._pushed_global_sec = 0.0
                            # 清理 Redis 进度，避免下一次推理被旧进度污染
                            try:
                                self._redis.delete(self._KEY_PROGRESS, self._KEY_SNAPSHOT, self._KEY_CLOCK)
                            except Exception:
                                pass
                            logger.info("ModuleSync 已重置，等待下一次推理触发")
                            continue
                        # 未追上：不发 done，落到下方正常帧率推送，按 1 倍速继续追赶

                    if not cycle_done:
                        self._push_events_up_to(global_sec)
            except Exception as e:
                logger.error(f"聚合循环中发生异常: {e}")

            elapsed = time.time() - loop_start
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _push_events_up_to(self, global_sec: float) -> None:
        """按帧率步长对齐推送：每次只推进 frame_interval 秒，取该窗口内的事件

        工业界流式同步做法：按固定时间步长分片，每个 batch 最多 1 帧视频，
        天然平滑，无需额外限速逻辑。global_sec 跳变时自动按帧率节奏追赶。
        """
        # 按帧率步长推进；global_sec 回退（新轮次）时 min() 自动重置基准
        target_sec = min(global_sec, self._pushed_global_sec + self.frame_interval)
        self._pushed_global_sec = target_sec

        events_to_push = []
        ids_to_delete = []

        try:
            # 增量读取：从上次位置之后开始（闭区间，跳过第一个已读过的）
            if self._last_stream_id == "0-0":
                entries = self._redis.xrange(self._KEY_EVENT_STREAM, min="-", max="+", count=200)
            else:
                entries = self._redis.xrange(self._KEY_EVENT_STREAM, min=self._last_stream_id, max="+", count=200)
                if entries and entries[0][0] == self._last_stream_id:
                    entries = entries[1:]  # 跳过上次读过的

            # 合并上次暂存的事件（local_sec > 旧 target_sec 的）
            all_events = list(self._pending_events)
            self._pending_events = []

            for entry_id, fields in entries:
                try:
                    ev = json.loads(fields["payload"])
                    local_sec = float(fields.get("local_sec", 0))
                    all_events.append((local_sec, ev, entry_id))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"解析事件失败: {e}")
                    ids_to_delete.append(entry_id)

            # 更新增量读取位置
            if entries:
                self._last_stream_id = entries[-1][0]

            all_events.sort(key=lambda x: x[0])

            for local_sec, ev, entry_id in all_events:
                if local_sec <= target_sec:
                    events_to_push.append(ev)
                    if entry_id:
                        ids_to_delete.append(entry_id)
                else:
                    # 未到时间的暂存到本地（entry_id=None 表示来自暂存）
                    self._pending_events.append((local_sec, ev, None))

            # 批量删除已处理事件
            if ids_to_delete:
                pipe = self._redis.pipeline()
                for entry_id in ids_to_delete:
                    pipe.xdel(self._KEY_EVENT_STREAM, entry_id)
                pipe.execute()

            if events_to_push:
                batch = {"globalSec": target_sec}
                source_counts = {}
                for ev in events_to_push:
                    source = ev.get("source", "unknown")
                    if source not in batch:
                        batch[source] = []
                        source_counts[source] = 0
                    batch[source].append({
                        "localSec": ev.get("localSec"),
                        "tag": ev.get("tag"),
                        "data": ev.get("data"),
                    })
                    source_counts[source] += 1
                self._do_push(batch)
                logger.debug(
                    f"批推送 global_sec={target_sec:.2f}/{global_sec:.2f}: "
                    + ", ".join(f"{s}={c}" for s, c in source_counts.items())
                )

        except Exception as e:
            logger.error(f"对齐并推送事件失败: {e}")

    def _flush_remaining_events(self) -> None:
        """强制刷新剩余的所有事件(打包为batch)"""
        try:
            entries = self._redis.xrange(self._KEY_EVENT_STREAM, min="-", max="+")
            if not entries:
                return
            context = self._get_context()
            events = []
            for entry_id, fields in entries:
                try:
                    ev = json.loads(fields["payload"])
                    ev["context"] = context
                    events.append(ev)
                except (json.JSONDecodeError, TypeError):
                    pass
                self._redis.xdel(self._KEY_EVENT_STREAM, entry_id)
            if events:
                batch = {"globalSec": 0}
                for ev in events:
                    source = ev.get("source", "unknown")
                    if source not in batch:
                        batch[source] = []
                    batch[source].append({"localSec": ev.get("localSec"), "tag": ev.get("tag"), "data": ev.get("data")})
                self._do_push(batch)
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
        """同步器推送单事件到前端 — 所有事件统一写入 Redis Stream，由对齐循环按 global_sec 推送

        前端看到的任何内容都必须经过 global_sec 对齐，禁止绕过对齐直接推送。
        done 哨兵不通过此接口，由 push_sentinel 单独处理。
        """
        if "localSec" not in data:
            logger.error(f"推送事件 '{event_type}' 缺少 localSec 字段，无法对齐，已丢弃")
            return
        try:
            ev = {"source": event_type, **data}
            self._event_counter += 1
            self._redis.xadd(self._KEY_EVENT_STREAM, {
                "local_sec": str(ev["localSec"]),
                "counter": str(self._event_counter),
                "payload": json.dumps(ev, ensure_ascii=False),
            })
            logger.debug(f"事件写入Stream: {event_type}, localSec={ev['localSec']}")
        except Exception as e:
            logger.error(f"同步器写入事件失败: {event_type}, {e}")

    def flush_remaining(self) -> None:
        """强制刷新 Stream 中所有剩余事件到前端（done 信号推送前调用，确保评估结果不丢）"""
        self._flush_remaining_events()

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
