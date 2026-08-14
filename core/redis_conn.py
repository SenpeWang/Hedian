"""Redis 连接池管理器 — 复用 TCP 连接，防止进程内创建过多独立连接套接字."""
import redis
import threading
from typing import Dict

_pool_lock = threading.Lock()
_pools: Dict[str, redis.ConnectionPool] = {}


def get_redis_client(host: str = "localhost",
                     port: int = 6379,
                     db: int = 0) -> redis.Redis:
    """返回复用连接池的 Redis 客户端."""
    key = f"{host}:{port}:{db}"
    with _pool_lock:
        if key not in _pools:
            _pools[key] = redis.ConnectionPool(host=host,
                                               port=port,
                                               db=db,
                                               decode_responses=True,
                                               max_connections=50)
    return redis.Redis(connection_pool=_pools[key])
