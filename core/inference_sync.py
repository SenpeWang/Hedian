"""模块同步器: 从 Redis 读取各 source 进度与事件, 按 global_sec 对齐后推送.

per-source 模型：进度按 source 记录，结束信号由模块退出时主动写。
global_sec = min(未结束且 expected 的 source 进度)；已结束 source 从 min 剔除、缺失视频帧用最后帧补全。
done = 所有 expected_sources 已上报结束。对齐完即发不限速，播放节奏交由前端 30fps 队列。
"""
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import redis

from core.inference_stream import (
    KEY_EVENT_STREAM,
    KEY_PROGRESS,
    KEY_SNAPSHOT,
    KEY_SOURCE_DONE,
    _fine_source,
    _SOURCE_CATEGORY,
)

logger = logging.getLogger("core.inference_sync")

# 轮询间隔（秒）：仅防 CPU 空转，不绑墙钟播放节奏
POLL_INTERVAL_SEC = 0.005
# 死锁兜底（秒）：所有 source 既没全完成、且 global_sec 长时间无推进时强制收尾，防僵尸 run
INFERENCE_DEADLOCK_TIMEOUT = 600.0


class InferenceSync:
    """模块同步器（per-source 对齐，去限速，主动结束信号驱动 done）."""

    def __init__(
        self,
        fps: float = 30.0,
        expected_sources: Optional[Set[str]] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        duration: float = 0.0,
        event_bus=None,
    ):
        """初始化同步器并清理上一轮遗留数据.

        Args:
            fps: 推理帧率, 供调用方参考.
            expected_sources: 参与 global_sec min 计算的 source 集合;
                None 时视为空集合(不产生 done).
            redis_host: Redis 主机地址.
            redis_port: Redis 端口.
            redis_db: 数据库编号.
            duration: 视频总时长(秒), 用于排除接近片尾的 source.
            event_bus: 跨进程消息总线; 提供时订阅 FLOW_STARTED / FLOW_ENDED.
        """
        self.fps = fps
        self.duration = duration
        self._event_bus = event_bus

        from core.redis_conn import get_redis_client
        self._redis = get_redis_client(host=redis_host, port=redis_port, db=redis_db)
        self._redis.ping()
        logger.info("InferenceSync (推理同步中间件) Redis 连接成功")
        self._subscribe_flow_events()

        # Redis keys
        self._progress_key = KEY_PROGRESS
        self._snapshot_key = KEY_SNAPSHOT
        self._event_stream_key = KEY_EVENT_STREAM
        # per-source 结束信号：field=source 名, value=final local_sec
        self._source_done_key = KEY_SOURCE_DONE

        # 主管理进程负责初始化时清理旧数据
        self._redis.delete(self._progress_key, self._snapshot_key,
                           self._event_stream_key, self._source_done_key)

        self._push_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        # expected_sources：参与 global_sec min 计算的 source 集合（per-source 粒度）
        self._expected_sources: Set[str] = expected_sources if expected_sources is not None else set()

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._event_counter = 0
        # 增量读取 Stream 的上次位置（闭区间，下次读取时跳过该 id）
        self._last_stream_id: str = "0-0"
        # 本地暂存 local_sec > 当前推送时间的事件，下次循环再处理
        self._pending_events: list = []
        # 已推送到的 global_sec（游标，用于日志/调试，不再做限速）
        self._pushed_global_sec: float = 0.0
        # 本轮推理是否已推送 done 哨兵
        self._cycle_done: bool = False
        # 死锁兜底：记录最后一次 global_sec 推进时间
        self._last_progress_ts: float = time.time()
        # 死锁兜底：最近一次观察到的 global_sec（用于识别真实推进）
        self._last_progress_sec: float = 0.0


    # ── 对外接口 ────────────────────────────────────────────────

    def set_push_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """设置推送回调函数.

        Args:
            callback: 收到对齐后的 batch 时调用, 入参为 batch 字典.
        """
        self._push_callback = callback

    def reset(self) -> None:
        """重置同步器状态，用于新一轮推理，避免旧结果混入当前 batch."""
        try:
            self._redis.delete(
                self._progress_key,
                self._snapshot_key,
                self._event_stream_key,
                self._source_done_key,
            )
            logger.info("InferenceSync 已清理 Redis 推理流相关 key（含 source_done）")
        except Exception as e:
            logger.error(f"InferenceSync 重置 Redis 失败: {e}")

        self._last_stream_id = "0-0"
        self._pending_events = []
        self._pushed_global_sec = 0.0
        self._event_counter = 0
        self._cycle_done = False
        self._last_progress_ts = time.time()
        self._last_progress_sec = 0.0
        logger.info("InferenceSync 对齐基准时间与状态已重置为 0.0 秒")

    def start(self) -> None:
        """启动事件对齐聚合线程."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True

        self._thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self._thread.start()
        logger.info("InferenceSync 线程启动")

    def stop(self) -> None:
        """停止事件对齐聚合线程."""
        if not self._running:
            return
        self._stop_event.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._flush_remaining_events()
        logger.info("InferenceSync 线程已停止")

    def push_sentinel(self) -> None:
        """推送终止信号(done 哨兵), 回调入参为 None."""
        if self._push_callback is not None:
            try:
                self._push_callback(None)
            except Exception as e:
                logger.warning(f"推送 done 哨兵回调失败: {e}")

    def push_display(self, event_type: str, data: Dict[str, Any]) -> None:
        """同步器推送单事件到前端.

        所有事件统一写入 Redis Stream, 由对齐循环按 global_sec 推送.
        前端看到的任何内容都必须经过 global_sec 对齐, 禁止绕过对齐直接推送.
        done 哨兵不通过此接口, 由 push_sentinel 单独处理.

        Args:
            event_type: 事件类型, 同时作为事件的 source.
            data: 事件载荷, 必须含 "localSec", 否则丢弃.
        """
        if "localSec" not in data:
            logger.error(f"推送事件 '{event_type}' 缺少 localSec 字段，无法对齐，已丢弃")
            return
        try:
            event = {"source": event_type, **data}
            self._event_counter += 1
            self._redis.xadd(self._event_stream_key, {
                "local_sec": str(event["localSec"]),
                "counter": str(self._event_counter),
                "payload": json.dumps(event, ensure_ascii=False),
            })
            logger.debug(f"事件写入Stream: {event_type}, localSec={event['localSec']}")
        except Exception as e:
            logger.error(f"同步器写入事件失败: {event_type}, {e}")

    def _on_flow_started(self, event: dict) -> None:
        """FLOW_STARTED → push_display flow_start 进 results:all(供前端事件流栏).

        Args:
            event: 总线消息字典, 业务载荷取其 "data" 字段.
        """
        payload = event.get("data", event)
        timestamp = float(payload.get("flow_start_sec", event.get("ts", 0)))
        self.push_display("flow_start", {
            "localSec": timestamp, "tag": "flow_start",
            "data": {"flow_id": payload.get("flow_id"), "flow_type": payload.get("flow_type"),
                     "flow_start_sec": timestamp}
        })

    def _on_flow_ended(self, event: dict) -> None:
        """FLOW_ENDED → push_display flow_end 进 results:all.

        Args:
            event: 总线消息字典, 业务载荷取其 "data" 字段.
        """
        payload = event.get("data", event)
        timestamp = float(payload.get("flow_end_sec", event.get("ts", 0)))
        self.push_display("flow_end", {
            "localSec": timestamp, "tag": "flow_end",
            "data": {"flow_id": payload.get("flow_id"), "flow_type": payload.get("flow_type"),
                     "flow_end_sec": timestamp, "flow_continue_sec": payload.get("flow_continue_sec", 0)}
        })

    def _subscribe_flow_events(self) -> None:
        """订阅 event_bus FLOW_STARTED/FLOW_ENDED, 转推 results:all 供前端事件流栏."""
        if not self._event_bus:
            return
        from core.event_bus import EventTopic
        self._event_bus.subscribe(EventTopic.FLOW_STARTED, self._on_flow_started)
        self._event_bus.subscribe(EventTopic.FLOW_ENDED, self._on_flow_ended)
        logger.info("InferenceSync 已订阅 FLOW_STARTED/FLOW_ENDED → flow_start/flow_end 推送")

    def flush_remaining(self) -> None:
        """强制刷新 Stream 中所有剩余事件到前端(done 信号推送前调用, 确保评估结果不丢)."""
        self._flush_remaining_events()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计状态.

        Returns:
            统计字典; 读取失败时返回空字典.
        """
        try:
            all_progress = self._redis.hgetall(self._progress_key)
            all_snapshots = self._redis.hgetall(self._snapshot_key)
            done_sources = self._get_done_sources()
            stream_len = self._redis.xlen(self._event_stream_key)
            return {
                "stream_size": stream_len,
                "global_sec": self._compute_global_sec(),
                "source_times": {field: float(value) for field, value in all_progress.items()},
                "done_sources": list(done_sources),
                "all_done": self._all_sources_done(),
                "module_snapshots": {module_name: json.loads(raw)
                                     for module_name, raw in all_snapshots.items()},
                "running": self._running,
                "fps": self.fps,
            }
        except Exception as e:
            logger.error(f"获取同步器状态失败: {e}")
            return {}

    # ── 私有辅助 ────────────────────────────────────────────────

    def _get_done_sources(self) -> Set[str]:
        """读取已上报结束信号的 source 集合.

        Returns:
            结束信号 Hash 的字段名集合.
        """
        try:
            return set(self._redis.hgetall(self._source_done_key).keys())
        except Exception as e:
            logger.error(f"读取 source_done 失败: {e}")
            return set()

    def _all_sources_done(self) -> bool:
        """判定所有预期 source 是否都已上报结束信号."""
        if not self._expected_sources:
            return False
        # source_done 字段为 "大类.细粒度"，按细粒度名与 expected_sources 比对
        done_fine = {_fine_source(field) for field in self._get_done_sources()}
        return self._expected_sources.issubset(done_fine)

    def _compute_global_sec(self) -> float:
        """全局时钟 = min(未结束且 expected 的 source 进度).

        Returns:
            全局时钟(秒); 无有效 source 或读取失败时返回 float("inf").
        """
        try:
            all_progress = self._redis.hgetall(self._progress_key)
            done_sources = self._get_done_sources()

            relevant = {}
            for source, value in all_progress.items():
                if source in done_sources:
                    continue
                # source 为 "大类.细粒度"; 用大类(_SOURCE_CATEGORY 映射)比对 expected
                # 修正: behavior.video_pop 细粒度名 video_pop 不匹配 expected(behavior)
                #       改用大类比对, 使 pop 视角进度计入 global_sec
                fine_source = _fine_source(source)
                category = _SOURCE_CATEGORY.get(fine_source, fine_source)
                if (self._expected_sources
                        and category not in self._expected_sources
                        and fine_source not in self._expected_sources):
                    continue
                try:
                    sec = float(value)
                except ValueError:
                    continue
                if self.duration > 0 and sec >= self.duration - 1.5:
                    continue
                relevant[source] = sec

            if not relevant:
                return float("inf")

            return min(relevant.values())
        except Exception as e:
            logger.error(f"计算全局时钟失败: {e}")
            return float("inf")

    def _get_module_snapshots(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模块的快照.

        Returns:
            {module_name: 快照字典}; 读取失败时返回空字典.
        """
        try:
            snapshots = self._redis.hgetall(self._snapshot_key)
            return {module_name: json.loads(raw) for module_name, raw in snapshots.items()}
        except Exception as e:
            logger.error(f"获取上下文快照失败: {e}")
            return {}

    def _reset_cycle(self) -> None:
        """本轮完成后重置游标与 Redis，准备等待下一次 /start."""
        self._cycle_done = True
        self._pushed_global_sec = 0.0
        self._last_stream_id = "0-0"
        self._pending_events = []
        try:
            self._redis.delete(self._progress_key, self._snapshot_key, self._source_done_key)
        except redis.RedisError as e:
            logger.warning(f"_reset_cycle 清理 Redis key 失败（不影响下一轮）: {e}")
        self._last_progress_ts = time.time()
        self._last_progress_sec = 0.0
        logger.info("InferenceSync 已重置本轮状态，等待下一次推理触发")

    def _is_deadlocked(self) -> bool:
        """判定是否触发死锁兜底.

        全部 source 均未结束且 global_sec 长时间无推进时返回 True.

        Returns:
            触发死锁兜底返回 True.
        """
        return time.time() - self._last_progress_ts > INFERENCE_DEADLOCK_TIMEOUT

    # ── Stream 读取 ─────────────────────────────────────────────

    def _read_stream_entries(self, count: Optional[int] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """从 Redis Stream 读取事件条目，自动处理 last_stream_id 跳过."""
        if self._last_stream_id == "0-0":
            entries = self._redis.xrange(self._event_stream_key, min="-", max="+", count=count)
        else:
            entries = self._redis.xrange(self._event_stream_key,
                                         min=self._last_stream_id, max="+", count=count)
            if entries and entries[0][0] == self._last_stream_id:
                entries = entries[1:]
        return entries

    def _parse_stream_entries(
        self, entries: List[Tuple[str, Dict[str, Any]]]
    ) -> Tuple[List[Tuple[float, Dict[str, Any], str]], List[str]]:
        """解析 Stream 条目为 (local_sec, event, entry_id) 三元组."""
        events: List[Tuple[float, Dict[str, Any], str]] = []
        ids_to_delete: List[str] = []
        for entry_id, fields in entries:
            try:
                event = json.loads(fields["payload"])
                local_sec = float(fields.get("local_sec", 0))
                events.append((local_sec, event, entry_id))
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"解析事件失败: {e}")
                ids_to_delete.append(entry_id)
        return events, ids_to_delete

    # ── Batch 构建与推送 ─────────────────────────────────────────

    def _build_batch(self, events: List[Dict[str, Any]], global_sec: float) -> Dict[str, Any]:
        """把事件列表聚合成前端 batch（视频去重、最后帧补全）.

        携带 sourceTimes: 各视角整体进度(秒), 供前端按视角算实时速度取 min 决定播放倍率.
        视角整体进度映射: front=tracker.tracking, pop=behavior.video_pop, voice=voice.voice.
        视频流走独立二进制通道(vis_stream→ws.send_bytes), 不进 results:all;
        此处只聚合结构化事件(progress/voice/tracking/gaze/flow/evaluation).

        Args:
            events: 已可推送的事件列表.
            global_sec: 本次 batch 对应的全局时钟.

        Returns:
            前端 batch 字典, 含 globalSec / totalDuration / sourceTimes 与各 source 事件组.
        """
        batch = {"globalSec": global_sec, "totalDuration": self.duration}
        # 各视角整体进度(供前端速率引擎)
        try:
            all_progress = self._redis.hgetall(self._progress_key)
            view_secs = {}
            for field, value in all_progress.items():
                try:
                    sec = float(value)
                except (ValueError, TypeError):
                    continue
                fine = _fine_source(field)
                # 视角整体进度归类: tracking/gaze/video_front -> front(取最大, 视角已产出位置)
                if fine in ("tracking", "gaze", "video_front"):
                    if sec > view_secs.get("front", -1): view_secs["front"] = sec
                elif fine in ("video_pop", "behavior"):
                    if sec > view_secs.get("pop", -1): view_secs["pop"] = sec
                elif fine == "voice":
                    view_secs["voice"] = sec
            batch["sourceTimes"] = view_secs
        except Exception as e:
            logger.warning(f"采集 sourceTimes 失败: {e}")

        for event in events:
            source = event.get("source", "unknown")
            item = {"localSec": event.get("localSec"), "tag": event.get("tag"), "data": event.get("data")}
            batch.setdefault(source, []).append(item)

        return batch

    def _push_event(self, event: Dict[str, Any]) -> None:
        """执行实际的回调推送.

        Args:
            event: 待推送的 batch 字典; 推送回调未设置时直接返回.
        """
        if self._push_callback is None:
            return
        try:
            self._push_callback(event)
        except Exception as e:
            logger.error(f"调用推送回调失败: {e}")

    # ── 主循环 ──────────────────────────────────────────────────

    def _aggregation_loop(self) -> None:
        """对齐推送循环：不限速对齐即发；done 由所有 source 结束驱动."""
        while not self._stop_event.is_set():
            try:
                global_sec = self._compute_global_sec()
                self._try_finish_or_push(global_sec)
            except Exception as e:
                logger.error(f"聚合循环中发生异常: {e}")
            time.sleep(POLL_INTERVAL_SEC)

    def _try_finish_or_push(self, global_sec: float) -> None:
        """流式直推模式：实时将已生成的结构化推理数据推向前端，由前端时序池进行视频时间帧同步渲染."""
        if self._cycle_done:
            return

        # 实时推送当前 Stream 中的所有事件到前端
        effective_sec = global_sec if global_sec != float("inf") else 999999.0
        self._push_events_up_to(effective_sec)

        # 死锁兜底口径: global_sec 真实增长才算推进(推理慢于实时属正常异步, 不应被强杀)
        if global_sec != float("inf") and global_sec > self._last_progress_sec:
            self._last_progress_sec = global_sec
            self._last_progress_ts = time.time()

        all_done = self._all_sources_done()

        # 所有 source 完成 → 收尾并 done
        if all_done:
            self._finish_cycle(effective_sec)
            return

        # 死锁兜底
        if self._is_deadlocked():
            logger.warning(f"推理死锁兜底触发（{INFERENCE_DEADLOCK_TIMEOUT:.0f}s 无推进），强制收尾推送 done")
            self._finish_cycle(effective_sec)

    def _finish_cycle(self, global_sec: float) -> None:
        """本轮推理收尾：推送最后一帧、刷新剩余事件、发送 done 哨兵、重置状态."""
        logger.info("所有 expected source 已上报结束信号，本轮推理完成，刷新剩余事件并推送 done")
        if global_sec != float("inf"):
            self._push_events_up_to(global_sec)
        self._flush_remaining_events()
        self.push_sentinel()
        self._reset_cycle()

    def _push_events_up_to(self, global_sec: float) -> None:
        """对齐推送：把 local_sec <= global_sec 的事件聚成一个 batch 推送."""
        try:
            entries = self._read_stream_entries(count=500)
            if not entries:
                return
            self._last_stream_id = entries[-1][0]

            parsed, ids_to_delete = self._parse_stream_entries(entries)
            ready, pending, more_ids = self._classify_events(parsed, global_sec)
            ids_to_delete.extend(more_ids)
            self._pending_events = pending
            self._delete_entries(ids_to_delete)
            # 滚动修剪已消费的高频视频帧与推理流，确保 Redis 极低内存水位（业务事件流保存在 module:events:* 永久保留）
            if self._event_counter % 20 == 0:
                try:
                    self._redis.xtrim(self._event_stream_key, maxlen=200000, approximate=True)  # 增大:保留整个推理期间数据，避免删未读条目导致前端数据缺失
                except Exception:
                    pass

            self._pushed_global_sec = global_sec
            if ready:
                self._push_event(self._build_batch(ready, global_sec))
        except Exception as e:
            logger.error(f"对齐并推送事件失败: {e}")

    def _classify_events(
        self, parsed: List[Tuple[float, Dict[str, Any], str]], global_sec: float
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[float, Dict[str, Any], Optional[str]]], List[str]]:
        """将 pending 事件与解析后的事件合并排序，按 global_sec 切分.

        Args:
            parsed: _parse_stream_entries 的待处理事件列表.
            global_sec: 切分阈值, local_sec 小于等于它的立即推送.

        Returns:
            三元组:
                ready: 可立即推送的事件列表.
                pending: 需留到下次循环的事件列表(entry_id 置 None).
                ids_to_delete: 已消费条目的 Stream ID 列表.
        """
        all_events = list(self._pending_events) + parsed
        all_events.sort(key=lambda item: item[0])
        self._pending_events = []

        ready: List[Dict[str, Any]] = []
        pending: List[Tuple[float, Dict[str, Any], Optional[str]]] = []
        ids_to_delete: List[str] = []

        for local_sec, event, entry_id in all_events:
            if local_sec <= global_sec:
                ready.append(event)
                if entry_id:
                    ids_to_delete.append(entry_id)
                continue
            pending.append((local_sec, event, None))

        return ready, pending, ids_to_delete

    def _flush_remaining_events(self) -> None:
        """强制刷新剩余的所有事件(打包为batch)，done 信号推送前调用."""
        try:
            entries = self._read_stream_entries()
            if not entries:
                return

            events = self._extract_events_with_context(entries)
            self._delete_entries([entry_id for entry_id, _ in entries])

            if not events:
                return
            events.sort(key=lambda event: float(event.get("localSec", 0)))
            self._push_event(self._build_batch(events, self._pushed_global_sec))
        except Exception as e:
            logger.error(f"清理剩余事件失败: {e}")

    def _extract_events_with_context(
        self, entries: List[Tuple[str, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """解析 Stream 条目并注入当前上下文快照.

        Args:
            entries: Stream 条目列表, 每项为 (entry_id, fields).

        Returns:
            已注入 "context" 字段的事件列表; 载荷无法解析的条目跳过.
        """
        snapshots = self._get_module_snapshots()
        events: List[Dict[str, Any]] = []
        for _, fields in entries:
            try:
                event = json.loads(fields["payload"])
                event["context"] = snapshots
                events.append(event)
            except (json.JSONDecodeError, TypeError):
                continue
        return events

    def _delete_entries(self, ids_to_delete: List[str]) -> None:
        """批量删除已消费 Stream 条目.

        Args:
            ids_to_delete: 待删除的 Stream ID 列表; 空列表时直接返回.
        """
        if not ids_to_delete:
            return
        try:
            pipe = self._redis.pipeline()
            for entry_id in ids_to_delete:
                pipe.xdel(self._event_stream_key, entry_id)
            pipe.execute()
        except Exception as e:
            logger.error(f"删除已消费 Stream 条目失败: {e}")
