"""
推理总线 — 供各业务算法模块进程使用，只负责向 Redis 写入进度和推理结果。
"""
import json
import logging
from typing import Any, Dict, Optional, Set

import redis

logger = logging.getLogger("core.inference_bus")


class InferenceBus:
    """推理总线（只写模式）"""

    def __init__(
        self,
        fps: float = 30.0,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        **kwargs
    ):
        self.fps = fps
        self._redis = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db,
            decode_responses=True, socket_connect_timeout=5,
        )
        self._redis.ping()
        logger.info("InferenceBus (推理总线) Redis 连接成功")

        # Redis keys
        self._KEY_PROGRESS = "inference:progress"
        self._KEY_SNAPSHOT = "inference:snapshot"
        self._KEY_EVENT_STREAM = "inference:results:all"
        self._KEY_CLOCK = "inference:global_sec"

        # 立即推送的事件类型
        self._immediate_types: Set[str] = {
            "progress", "video_start",
        }

        self._event_counter = 0

    def update_module_time(self, module_name: str, sec: float) -> None:
        """更新模块的当前进度"""
        try:
            self._redis.hset(self._KEY_PROGRESS, module_name, str(sec))
        except Exception as e:
            logger.error(f"更新模块进度失败 {module_name}: {e}")

    def update_module_snapshot(self, module_name: str, snapshot: Dict[str, Any]) -> None:
        """更新模块的状态快照"""
        try:
            self._redis.hset(self._KEY_SNAPSHOT, module_name,
                             json.dumps(snapshot, ensure_ascii=False))
        except Exception as e:
            logger.error(f"更新模块快照失败 {module_name}: {e}")

    def remove_module(self, module_name: str) -> None:
        """从进度追踪中移除模块"""
        try:
            self._redis.hdel(self._KEY_PROGRESS, module_name)
        except Exception as e:
            logger.error(f"移除模块失败 {module_name}: {e}")

    def _get_context(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模块的最新快照"""
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

    def push_display(self, event_type: str, data: Dict[str, Any]) -> None:
        """向 Redis 推送模块推理事件"""
        ev = {"source": event_type, **data}
        logger.debug(f"InferenceBus 收到事件: source={event_type}, localSec={data.get('localSec', 'N/A')}")

        try:
            if event_type in self._immediate_types:
                if "localSec" in ev:
                    ev["context"] = self._get_context()
                self._event_counter += 1
                local_sec = ev.get("localSec", 0)
                self._redis.xadd(self._KEY_EVENT_STREAM, {
                    "local_sec": str(local_sec),
                    "counter": str(self._event_counter),
                    "payload": json.dumps(ev, ensure_ascii=False),
                })
                return

            if "localSec" not in ev:
                logger.error(f"时间对齐事件类型 '{event_type}' 必须包含 'localSec' 字段")
                return

            # 普通事件直接写入 Stream
            self._event_counter += 1
            local_sec = ev["localSec"]
            self._redis.xadd(self._KEY_EVENT_STREAM, {
                "local_sec": str(local_sec),
                "counter": str(self._event_counter),
                "payload": json.dumps(ev, ensure_ascii=False),
            })
        except Exception as e:
            logger.error(f"推送推理事件失败: {event_type}, {e}")

    def start(self) -> None:
        """只写模式下的启动占位"""
        logger.info("InferenceBus 启动")

    def stop(self) -> None:
        """只写模式下的停止占位"""
        logger.info("InferenceBus 停止")

    def push_sentinel(self) -> None:
        """推送终止占位"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """获取统计状态"""
        return {"writer_mode": True, "fps": self.fps}

    def clear(self) -> None:
        """清理数据"""
        pass
