"""
消息总线 — 基于 Redis Stream 的跨进程通信

消息格式:
  {"type": str, "data": dict, "ts": float}

用法:
  event_bus = EventBus()
  event_bus.start()
  event_bus.subscribe("voice.intent", my_callback)
  event_bus.publish("voice.intent", {"text": "..."}, ts=1.5)
  event_bus.stop()
"""
import json
import threading
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

import redis

logger = logging.getLogger("core.event_bus")


class EventTopic:
    """消息类型常量（同时也是 Redis Stream key）"""

    # Voice -> EventBus
    VOICE_KEY_MOMENT = "voice.key_moment"

    # Tracker -> EventBus
    TRACKER_PROXIMITY = "tracker.proximity"
    TRACKER_HEADCOUNT = "tracker.headcount"

    # Behavior -> EventBus
    BEHAVIOR_HAND_RAISED = "behavior.hand_raised"

    # Gaze -> EventBus
    GAZE_ATTENTION = "gaze.attention"
    GAZE_ALERT = "gaze.alert"

    # Rules -> EventBus
    FLOW_STARTED = "flow.started"
    FLOW_ENDED = "flow.ended"
    RULE_KEY_MOMENT = "rule.key_moment"

    # Evaluation -> All modules：通知各模块立即保存 key_moments
    SAVE_KEY_MOMENTS = "save.key_moments"


class EventBus:
    """基于 Redis Stream 的发布/订阅消息总线"""

    # Stream key 前缀
    STREAM_PREFIX = "module:events:"

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379,
                 redis_db: int = 0, max_workers: int = 4, consumer_name: str = None,
                 **kwargs):
        """
        初始化消息总线

        Args:
            redis_host: Redis 服务器地址
            redis_port: Redis 端口
            redis_db: Redis 数据库编号
            max_workers: 最大并发分发线程数
            consumer_name: 消费者名称（每个进程应该唯一）
        """
        from core.redis_conn import get_redis_client
        self._redis = get_redis_client(host=redis_host, port=redis_port, db=redis_db)
        self._consumer_name = consumer_name or f"consumer_{int(time.time() * 1000)}"
        # 每进程独立消费组实现跨进程广播
        self._consumer_group = f"{self._consumer_name}_group"
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._listener: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._message_count = 0

        # 测试连接
        try:
            self._redis.ping()
            logger.info(f"Redis 连接成功: {redis_host}:{redis_port}/{redis_db}, 消费组: {self._consumer_group}")
        except redis.ConnectionError as e:
            logger.error(f"Redis 连接失败: {e}")
            raise

    def _get_stream_key(self, msg_type: str) -> str:
        """获取 Stream key"""
        return f"{self.STREAM_PREFIX}{msg_type}"

    def _ensure_consumer_group(self, stream_key: str) -> None:
        """确保消费者组存在"""
        try:
            self._redis.xgroup_create(
                stream_key, self._consumer_group, id="0", mkstream=True
            )
            logger.debug(f"创建消费者组: {stream_key} / {self._consumer_group}")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def publish(self, msg_type: str, data: dict, ts: float = 0.0) -> None:
        """
        发布消息到 Redis Stream

        Args:
            msg_type: 消息类型
            data: 消息数据
            ts: 时间戳
        """
        msg = {"type": msg_type, "data": data, "ts": ts}
        stream_key = self._get_stream_key(msg_type)

        try:
            payload = json.dumps(msg, ensure_ascii=False)
            # XADD 写入 Stream，消息持久化
            self._redis.xadd(stream_key, {"payload": payload}, maxlen=10000)
            self._message_count += 1
            logger.debug(f"发布消息: {msg_type}")
        except Exception as e:
            logger.error(f"发布消息失败: {msg_type}, {e}")

    def subscribe(self, msg_type: str, callback: Callable) -> None:
        """
        订阅消息类型

        Args:
            msg_type: 消息类型
            callback: 回调函数
        """
        with self._lock:
            if msg_type not in self._subscribers:
                self._subscribers[msg_type] = []
            self._subscribers[msg_type].append(callback)

        # 确保消费者组存在
        stream_key = self._get_stream_key(msg_type)
        self._ensure_consumer_group(stream_key)

        # 若已 start 但 listener 未运行，现在补启动
        if self._running and (self._listener is None or not self._listener.is_alive()):
            self._listener = threading.Thread(target=self._listen_loop, daemon=True)
            self._listener.start()
            logger.info(f"延迟启动 listener: {msg_type}")

        logger.debug(f"订阅消息: {msg_type}")

    def unsubscribe(self, msg_type: str, callback: Callable) -> None:
        """
        取消订阅

        Args:
            msg_type: 消息类型
            callback: 回调函数
        """
        with self._lock:
            if msg_type in self._subscribers:
                self._subscribers[msg_type] = [
                    cb for cb in self._subscribers[msg_type] if cb != callback
                ]
        logger.debug(f"取消订阅: {msg_type}")

    def start(self) -> None:
        """启动 listener 线程"""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()

        with self._lock:
            channels = list(self._subscribers.keys())

        if channels:
            # 为每个频道创建消费者组
            for ch in channels:
                self._ensure_consumer_group(self._get_stream_key(ch))

            self._listener = threading.Thread(
                target=self._listen_loop, daemon=True
            )
            self._listener.start()
            logger.info(f"消息总线启动，订阅频道: {channels}, 消费者: {self._consumer_name}")
        else:
            logger.info("消息总线启动（无订阅频道）")

    def _listen_loop(self) -> None:
        """监听循环：从所有订阅的 Stream 读取消息（动态支持运行时新增订阅）"""
        while not self._stop_event.is_set():
            # 每次循环动态读取订阅列表，支持运行时 subscribe 新 topic
            with self._lock:
                stream_keys = [self._get_stream_key(ch) for ch in self._subscribers.keys()]

            if not stream_keys:
                time.sleep(0.5)
                continue

            # 从最新消息开始读取
            last_ids = {key: ">" for key in stream_keys}

            try:
                # XREADGROUP 阻塞读取，超时 1 秒
                results = self._redis.xreadgroup(
                    self._consumer_group,
                    self._consumer_name,
                    last_ids,
                    count=100,
                    block=1000,  # 1 秒超时
                )

                if not results:
                    continue

                for stream_key, messages in results:
                    # 从 stream_key 中提取 msg_type
                    msg_type = stream_key.replace(self.STREAM_PREFIX, "")

                    for entry_id, fields in messages:
                        try:
                            payload = fields.get("payload", "{}")
                            msg = json.loads(payload)
                            msg_type_actual = msg.get("type", msg_type)

                            with self._lock:
                                callbacks = list(self._subscribers.get(msg_type_actual, []))

                            for cb in callbacks:
                                self._executor.submit(self._safe_call, cb, msg, msg_type_actual)

                            # 确认消息已处理
                            self._redis.xack(stream_key, self._consumer_group, entry_id)

                        except (json.JSONDecodeError, TypeError) as e:
                            logger.error(f"解析消息失败: {stream_key}, {e}")
                            # 确认消息，避免重复处理
                            self._redis.xack(stream_key, self._consumer_group, entry_id)

            except redis.exceptions.ConnectionError as e:
                logger.error(f"Redis 连接断开: {e}")
                time.sleep(1)
            except Exception as e:
                err_msg = str(e)
                # NOGROUP 错误：消费者组不存在，自动重建
                if "NOGROUP" in err_msg:
                    for stream_key in stream_keys:
                        try:
                            self._ensure_consumer_group(stream_key)
                        except Exception:
                            pass
                    # 短暂休眠避免日志刷屏
                    time.sleep(0.5)
                else:
                    logger.error(f"监听循环异常: {e}")
                    time.sleep(0.1)

    def _safe_call(self, cb: Callable, msg: dict, msg_type: str) -> None:
        """安全调用订阅者回调"""
        try:
            cb(msg)
        except Exception as e:
            logger.error(f"订阅者处理 {msg_type} 失败: {e}", exc_info=True)

    def stop(self) -> None:
        """停止 listener 线程"""
        if not self._running:
            return
        self._stop_event.set()
        self._running = False

        if self._listener and self._listener.is_alive():
            self._listener.join(timeout=3.0)
        self._executor.shutdown(wait=False)
        logger.info(f"消息总线停止，共处理 {self._message_count} 条消息")

    def get_stats(self) -> dict:
        """获取总线统计"""
        with self._lock:
            sub_info = {k: len(v) for k, v in self._subscribers.items()}

        # 获取 Stream 信息
        stream_info = {}
        for msg_type in self._subscribers.keys():
            stream_key = self._get_stream_key(msg_type)
            try:
                info = self._redis.xinfo_stream(stream_key)
                stream_info[msg_type] = {
                    "length": info.get("length", 0),
                    "first_entry": info.get("first-entry"),
                    "last_entry": info.get("last-entry"),
                }
            except Exception:
                stream_info[msg_type] = {"length": 0}

        return {
            "running": self._running,
            "subscribers": sub_info,
            "streams": stream_info,
            "message_count": self._message_count,
            "consumer_name": self._consumer_name,
        }
