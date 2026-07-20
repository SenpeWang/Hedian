"""
Redis 连接池管理器 — 复用 TCP 连接，防止进程内创建过多独立连接套接字。
"""
import redis
import threading
from typing import Dict

_pool_lock = threading.Lock()
_pools: Dict[str, redis.ConnectionPool] = {}

def get_redis_client(host: str = "localhost", port: int = 6379, db: int = 0) -> redis.Redis:
    """
    获取使用共享连接池的 Redis 客户端。

    Args:
        host: Redis 主机地址
        port: Redis 端口
        db: Redis 数据库索引

    Returns:
        redis.Redis 客户端实例
    """
    key = f"{host}:{port}:{db}"
    with _pool_lock:
        if key not in _pools:
            _pools[key] = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                max_connections=50
            )
    return redis.Redis(connection_pool=_pools[key])
