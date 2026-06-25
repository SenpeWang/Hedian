"""
Redis 队列适配器

提供与 queue.Queue 相同接口的 Redis List 实现，
用于跨进程帧队列传输。
"""
import redis
import logging

logger = logging.getLogger("core.redis_queue")


class RedisQueue:
    """
    基于 Redis List 的队列

    接口与 queue.Queue 兼容：
    - put_nowait(item): 入队
    - get_nowait(): 出队（不阻塞）
    - get(timeout): 出队（阻塞等待）
    """

    def __init__(self, key: str, redis_host: str = "localhost",
                 redis_port: int = 6379, redis_db: int = 0,
                 maxsize: int = 0, redis_client=None):
        """
        初始化 Redis 队列

        Args:
            key: Redis List key
            redis_host: Redis 地址
            redis_port: Redis 端口
            redis_db: Redis 数据库
            maxsize: 最大长度（0=无限制）
            redis_client: 复用已有的 Redis 连接
        """
        self._key = key
        self._maxsize = maxsize
        if redis_client:
            self._redis = redis_client
        else:
            self._redis = redis.Redis(
                host=redis_host, port=redis_port, db=redis_db,
                decode_responses=False,
            )

    def put_nowait(self, item) -> None:
        """入队（不阻塞）"""
        if self._maxsize > 0:
            # 检查长度，超过则丢弃最旧的
            length = self._redis.llen(self._key)
            if length >= self._maxsize:
                self._redis.rpop(self._key)
        self._redis.lpush(self._key, item)

    def get_nowait(self):
        """出队（不阻塞）"""
        result = self._redis.rpop(self._key)
        if result is None:
            from queue import Empty
            raise Empty()
        return result

    def get(self, timeout: float = None):
        """
        出队（阻塞等待）

        Args:
            timeout: 超时时间（秒），None=永久等待

        Returns:
            队列中的数据

        Raises:
            queue.Empty: 超时后队列仍为空
        """
        if timeout is None:
            # 永久等待
            result = self._redis.brpop(self._key, timeout=0)
            return result[1] if result else None
        else:
            result = self._redis.brpop(self._key, timeout=int(timeout))
            if result is None:
                from queue import Empty
                raise Empty()
            return result[1]

    def empty(self) -> bool:
        """检查队列是否为空"""
        return self._redis.llen(self._key) == 0

    def qsize(self) -> int:
        """获取队列长度"""
        return self._redis.llen(self._key)
