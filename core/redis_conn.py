"""Redis 连接池管理器.

按 host/port/db 组合复用 TCP 连接池, 避免进程内频繁创建独立套接字.
"""

import threading
from typing import Dict

import redis

_pool_lock = threading.Lock()
_pools: Dict[str, redis.ConnectionPool] = {}


def get_redis_client(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
) -> redis.Redis:
    """返回复用连接池的 Redis 客户端.

    同一 host/port/db 组合只创建一个 ConnectionPool; 统一以 decode_responses=True
    建立连接, 使读回的字段直接为 str 而非 bytes.

    Args:
        host: Redis 主机地址.
        port: Redis 端口.
        db: 数据库编号.

    Returns:
        绑定连接池的 Redis 客户端.
    """
    key = f"{host}:{port}:{db}"
    with _pool_lock:
        if key not in _pools:
            _pools[key] = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                max_connections=50,
            )
    return redis.Redis(connection_pool=_pools[key])
