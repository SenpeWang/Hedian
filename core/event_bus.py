"""基于 Redis Stream 的跨进程发布订阅消息总线.

消息格式:
  {"type": str, "data": dict, "ts": float}

用法:
  event_bus = EventBus()
  event_bus.start()
  event_bus.subscribe("voice.intent", my_callback)
  event_bus.publish("voice.intent", {"text": "..."}, timestamp=1.5)
  event_bus.stop()
"""
import json
import logging
import threading
import time
from typing import Callable, Dict, List, Optional

import redis

logger = logging.getLogger("core.event_bus")


class EventTopic:
    """消息类型常量(同时也是 Redis Stream key)."""

    # Voice -> EventBus
    VOICE_KEY_MOMENT = "voice.key_moment"

    # Tracker -> EventBus
    TRACKER_PROXIMITY = "tracker.proximity"
    TRACKER_HEADCOUNT = "tracker.headcount"

    # Behavior -> EventBus
    BEHAVIOR_HAND_RAISED = "behavior.hand_raised"
    BEHAVIOR_FINGER_SCREEN = "behavior.finger_screen"
    BEHAVIOR_FINGER_FILE = "behavior.finger_file"

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
    """基于 Redis Stream 的发布/订阅消息总线."""

    STREAM_PREFIX = "module:events:"

    def __init__(self,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 consumer_name: Optional[str] = None,
                 **kwargs):
        """初始化总线并验证 Redis 连通性.

        每进程使用独立消费组实现跨进程广播.

        Args:
            redis_host: Redis 主机地址.
            redis_port: Redis 端口.
            redis_db: 数据库编号.
            consumer_name: 消费者名; None 时按当前毫秒时间生成.
            **kwargs: 预留扩展参数.
        """
        from core.redis_conn import get_redis_client
        self._redis = get_redis_client(host=redis_host,
                                       port=redis_port,
                                       db=redis_db)
        self._consumer_name = consumer_name or f"consumer_{int(time.time() * 1000)}"
        # 每进程独立消费组实现跨进程广播
        self._consumer_group = f"{self._consumer_name}_group"
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._listener: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._message_count = 0

        try:
            self._redis.ping()
            logger.info(
                f"Redis 连接成功: {redis_host}:{redis_port}/{redis_db}, 消费组: {self._consumer_group}"
            )
        except redis.ConnectionError as e:
            logger.error(f"Redis 连接失败: {e}")
            raise

    def _get_stream_key(self, msg_type: str) -> str:
        """拼接 Redis Stream key.

        Args:
            msg_type: 消息类型.

        Returns:
            以 STREAM_PREFIX 为前缀的 Stream key.
        """
        return f"{self.STREAM_PREFIX}{msg_type}"

    def _ensure_consumer_group(self, stream_key: str) -> None:
        """确保消费组存在.

        组已存在时 XGROUP CREATE 抛 BUSYGROUP, 属预期情况, 直接忽略.

        Args:
            stream_key: Redis Stream key.
        """
        try:
            self._redis.xgroup_create(stream_key,
                                      self._consumer_group,
                                      id="0",
                                      mkstream=True)
            logger.debug(f"创建消费者组: {stream_key} / {self._consumer_group}")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def publish(self, msg_type: str, data: dict, timestamp: float = 0.0) -> None:
        """发布一条消息.

        Args:
            msg_type: 消息类型, 决定写入哪个 Stream.
            data: 业务载荷.
            timestamp: 消息时间戳; 0.0 表示未提供.
        """
        event = {"type": msg_type, "data": data, "ts": timestamp}
        stream_key = self._get_stream_key(msg_type)

        try:
            payload = json.dumps(event, ensure_ascii=False)
            self._redis.xadd(stream_key, {"payload": payload}, maxlen=10000)
            self._message_count += 1
            logger.debug(f"发布消息: {msg_type}")
        except (redis.RedisError, TypeError, ValueError) as e:
            logger.error(f"发布消息失败: {msg_type}, {e}")

    def subscribe(self, msg_type: str, callback: Callable) -> None:
        """订阅消息类型.

        已 start 但 listener 未运行时会补启动监听线程.

        Args:
            msg_type: 要订阅的消息类型.
            callback: 收到消息时的回调, 入参为消息字典.
        """
        with self._lock:
            if msg_type not in self._subscribers:
                self._subscribers[msg_type] = []
            self._subscribers[msg_type].append(callback)

        stream_key = self._get_stream_key(msg_type)
        self._ensure_consumer_group(stream_key)

        # 若已 start 但 listener 未运行，现在补启动
        if self._running and (self._listener is None
                              or not self._listener.is_alive()):
            self._listener = threading.Thread(target=self._listen_loop,
                                              daemon=True)
            self._listener.start()
            logger.info(f"延迟启动 listener: {msg_type}")

        logger.debug(f"订阅消息: {msg_type}")

    def start(self) -> None:
        """启动监听线程.

        无订阅频道时只置运行标记, 不创建线程.
        """
        if self._running:
            return
        self._running = True
        self._stop_event.clear()

        with self._lock:
            subscribed_topics = list(self._subscribers.keys())

        if subscribed_topics:
            for msg_type in subscribed_topics:
                self._ensure_consumer_group(self._get_stream_key(msg_type))

            self._listener = threading.Thread(target=self._listen_loop,
                                              daemon=True)
            self._listener.start()
            logger.info(f"消息总线启动，订阅频道: {subscribed_topics}, 消费者: {self._consumer_name}")
        else:
            logger.info("消息总线启动（无订阅频道）")

    def _listen_loop(self) -> None:
        """监听循环：动态读取订阅列表，支持运行时新增订阅."""
        while not self._stop_event.is_set():
            with self._lock:
                stream_keys = [
                    self._get_stream_key(msg_type)
                    for msg_type in self._subscribers.keys()
                ]

            if not stream_keys:
                time.sleep(0.5)
                continue

            last_ids = {key: ">" for key in stream_keys}

            try:
                # XREADGROUP 阻塞读取，超时 1 秒
                stream_entries = self._redis.xreadgroup(
                    self._consumer_group,
                    self._consumer_name,
                    last_ids,
                    count=100,
                    block=1000,  # 1 秒超时
                )

                if not stream_entries:
                    continue

                for stream_key, messages in stream_entries:
                    msg_type = stream_key.replace(self.STREAM_PREFIX, "")

                    for entry_id, fields in messages:
                        try:
                            payload = fields.get("payload", "{}")
                            event = json.loads(payload)
                            actual_msg_type = event.get("type", msg_type)

                            with self._lock:
                                callbacks = list(
                                    self._subscribers.get(actual_msg_type, []))

                            # 先执行回调全部成功后 xack：避免"先 ack 后回调"在进程退出/回调失败时事件永久丢失
                            for callback in callbacks:
                                self._safe_call(callback, event, actual_msg_type)

                            self._redis.xack(stream_key, self._consumer_group,
                                             entry_id)

                        except (json.JSONDecodeError, TypeError) as e:
                            logger.error(f"解析消息失败: {stream_key}, {e}")
                            # 确认消息，避免重复处理
                            self._redis.xack(stream_key, self._consumer_group,
                                             entry_id)

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
                        except redis.RedisError as e:
                            logger.warning(f"重建消费者组失败: {stream_key}, {e}")
                    # 短暂休眠避免日志刷屏
                    time.sleep(0.5)
                else:
                    logger.error(f"监听循环异常: {e}")
                    time.sleep(0.1)

    def _safe_call(self, callback: Callable, event: dict, msg_type: str) -> None:
        """调用订阅者回调, 吞掉异常并记录日志.

        Args:
            callback: 订阅者回调.
            event: 待投递的消息字典.
            msg_type: 消息类型, 仅用于日志.
        """
        try:
            callback(event)
        except Exception as e:
            logger.error(f"订阅者处理 {msg_type} 失败: {e}", exc_info=True)

    def stop(self) -> None:
        """停止总线并等待监听线程退出."""
        if not self._running:
            return
        self._stop_event.set()
        self._running = False

        if self._listener and self._listener.is_alive():
            self._listener.join(timeout=3.0)
        logger.info(f"消息总线停止，共处理 {self._message_count} 条消息")
