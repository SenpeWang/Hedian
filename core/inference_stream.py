"""推理总线写入端.

供各业务算法模块进程使用, 只负责向 Redis 写入进度和推理结果.
"""
import json
import logging
from typing import Any, Dict, Set

logger = logging.getLogger("core.inference_stream")


# source → 大类(模块) 映射：进度/结束信号字段统一写成 "大类.细粒度"
# 大类便于按模块聚合查询进度，细粒度仍可取 min 算全局时钟。
# 例：tracker 模块写 tracker.tracking / tracker.video_front；gaze 虽由 tracker
# 代写进度，但作为独立模态归属 gaze 大类 → gaze.gaze。
_SOURCE_CATEGORY: Dict[str, str] = {
    "voice": "voice",
    "tracking": "tracker",
    "video_front": "tracker",
    "gaze": "gaze",
    "video_pop": "behavior",
    "behavior": "behavior",
}


def _progress_field(source: str) -> str:
    """返回 "大类.细粒度" 形式的进度字段名.

    Args:
        source: 细粒度 source 名; 未登记时退化为 {source}.{source}.

    Returns:
        进度字段名, 如 tracker.tracking.
    """
    category = _SOURCE_CATEGORY.get(source, source)
    return f"{category}.{source}"


def _fine_source(field: str) -> str:
    """从 "大类.细粒度" 字段名提取细粒度 source 名.

    各 source 的细粒度名全局唯一, 故按细粒度名比对即可正确归属大类.

    Args:
        field: "大类.细粒度" 字段名.

    Returns:
        细粒度 source 名; 字段无 "." 前缀时原样返回.
    """
    return field.split(".", 1)[-1]


# 推理流 Redis key（写入端 / 同步中间件 / 数据提取器 / 模块基类共用，集中定义避免字符串散落）
KEY_PROGRESS = "inference:progress"
KEY_SNAPSHOT = "inference:snapshot"
KEY_EVENT_STREAM = "inference:results:all"
KEY_SOURCE_DONE = "inference:source_done"


class InferenceStream:
    """推理总线(只写模式)."""

    def __init__(
        self,
        fps: float = 30.0,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        **kwargs
    ):
        """初始化写入端并验证 Redis 连通性.

        Args:
            fps: 推理帧率, 供调用方参考.
            redis_host: Redis 主机地址.
            redis_port: Redis 端口.
            redis_db: 数据库编号.
            **kwargs: 预留扩展参数.
        """
        self.fps = fps
        from core.redis_conn import get_redis_client
        self._redis = get_redis_client(host=redis_host, port=redis_port, db=redis_db)
        self._redis.ping()
        logger.info("InferenceStream (推理流写入端) Redis 连接成功")

        # Redis keys
        self._progress_key = KEY_PROGRESS
        self._snapshot_key = KEY_SNAPSHOT
        self._event_stream_key = KEY_EVENT_STREAM
        self._clock_key = "inference:global_sec"
        # per-source 结束信号：field=source, value=final local_sec（模块退出时写入，供中间件剔除已结束 source）
        self._source_done_key = KEY_SOURCE_DONE

        # 立即推送的事件类型
        self._immediate_types: Set[str] = {
            "progress", "video_start",
        }

        self._event_counter = 0
        # 各 source 最近 local_sec，模块退出时用于批量上报结束信号
        self._source_last_sec: Dict[str, float] = {}

    def update_module_time(self, source: str, sec: float) -> None:
        """更新某个 source 的当前进度(per-source 粒度, 供对齐中间件 min 计算).

        字段以 "大类.细粒度" 形式写入(如 tracker.tracking), 便于按模块聚合查询.

        Args:
            source: 细粒度 source 名.
            sec: 当前进度(local_sec).
        """
        try:
            self._redis.hset(self._progress_key, _progress_field(source), str(sec))
        except Exception as e:
            logger.error(f"更新 source 进度失败 {source}: {e}")

    def update_module_snapshot(self, module_name: str, snapshot: Dict[str, Any]) -> None:
        """更新模块的状态快照.

        Args:
            module_name: 模块名, 作为 Hash 字段名.
            snapshot: 快照内容, 序列化为 JSON 存储.
        """
        try:
            self._redis.hset(self._snapshot_key, module_name,
                             json.dumps(snapshot, ensure_ascii=False))
        except Exception as e:
            logger.error(f"更新模块快照失败 {module_name}: {e}")

    def remove_module(self, module_name: str) -> None:
        """从进度追踪中移除模块.

        Args:
            module_name: 模块名, 作为进度 Hash 的字段名.
        """
        try:
            self._redis.hdel(self._progress_key, module_name)
        except Exception as e:
            logger.error(f"移除模块失败 {module_name}: {e}")

    def _get_module_snapshots(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模块的最新快照.

        快照无法解析时该模块记为空字典, 保证返回结构稳定.

        Returns:
            {module_name: 快照字典}.
        """
        try:
            snapshots = self._redis.hgetall(self._snapshot_key)
            result = {}
            for module_name, raw in snapshots.items():
                try:
                    result[module_name] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    result[module_name] = {}
            return result
        except Exception as e:
            logger.error(f"获取上下文快照失败: {e}")
            return {}

    def push_display(self, event_type: str, data: Dict[str, Any]) -> None:
        """向 Redis 推送模块推理事件.

        立即推送类型(progress / video_start)直接入 Stream; 其余类型必须携带
        localSec, 由对齐中间件按全局时钟排序后推送.

        Args:
            event_type: 事件类型, 同时作为事件的 source.
            data: 事件载荷, 时间对齐事件须含 "localSec".
        """
        event = {"source": event_type, **data}
        logger.debug(f"InferenceStream 收到事件: source={event_type}, localSec={data.get('localSec', 'N/A')}")

        try:
            if event_type in self._immediate_types:
                if "localSec" in event:
                    event["context"] = self._get_module_snapshots()
                self._event_counter += 1
                local_sec = event.get("localSec", 0)
                self._redis.xadd(self._event_stream_key, {
                    "local_sec": str(local_sec),
                    "counter": str(self._event_counter),
                    "payload": json.dumps(event, ensure_ascii=False),
                })
                return

            if "localSec" not in event:
                logger.error(f"时间对齐事件类型 '{event_type}' 必须包含 'localSec' 字段")
                return

            # 记录该 source 的最近 local_sec，供模块退出时上报结束信号
            self._source_last_sec[event_type] = float(event["localSec"])

            # 普通事件直接写入 Stream
            self._event_counter += 1
            local_sec = event["localSec"]
            self._redis.xadd(self._event_stream_key, {
                "local_sec": str(local_sec),
                "counter": str(self._event_counter),
                "payload": json.dumps(event, ensure_ascii=False),
            })
        except Exception as e:
            logger.error(f"推送推理事件失败: {event_type}, {e}")

    def mark_source_done(self, source: str, final_sec: float) -> None:
        """标记单个 source 推理结束(写入最终 local_sec).

        Args:
            source: 细粒度 source 名.
            final_sec: 该 source 的最终 local_sec.
        """
        try:
            self._redis.hset(self._source_done_key, _progress_field(source), str(final_sec))
            logger.info(f"source '{source}' 推理结束，final local_sec={final_sec:.2f}")
        except Exception as e:
            logger.error(f"标记 source 结束失败 {source}: {e}")

    def mark_sources_done(self, source_final_map: Dict[str, float]) -> None:
        """批量标记多个 source 推理结束.

        模块退出时一次性上报它产出的所有 source, 用 pipeline 减少往返.

        Args:
            source_final_map: {细粒度 source 名: 最终 local_sec}.
        """
        if not source_final_map:
            return
        try:
            pipe = self._redis.pipeline()
            for source, final_sec in source_final_map.items():
                pipe.hset(self._source_done_key, _progress_field(source), str(final_sec))
            pipe.execute()
            logger.info(f"批量标记 source 结束: {source_final_map}")
        except Exception as e:
            logger.error(f"批量标记 source 结束失败: {e}")

    def mark_all_sources_done(self) -> None:
        """把当前进程已登记的所有 source 的最后 local_sec 全部上报为结束.

        作为模块退出时的兜底路径.
        """
        if not self._source_last_sec:
            return
        self.mark_sources_done(dict(self._source_last_sec))

    def mark_owned_done(self, sources: set, default_sec: float = 0.0) -> None:
        """上报指定 source 集合的结束信号.

        Args:
            sources: 需上报结束的 source 集合.
            default_sec: 从未推送过事件的 source 使用的兜底 local_sec,
                防止全程无事件导致 done 永不触发.
        """
        if not sources:
            return
        final_map = {source: self._source_last_sec.get(source, default_sec) for source in sources}
        self.mark_sources_done(final_map)

    def start(self) -> None:
        """只写模式下的启动占位."""
        logger.info("InferenceStream 启动")

    def stop(self) -> None:
        """停止并上报结束信号(兜底, 主路径靠 BaseModule 退出时主动 mark)."""
        self.mark_all_sources_done()
        logger.info("InferenceStream 停止")
