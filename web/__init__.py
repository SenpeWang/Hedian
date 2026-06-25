"""
前端层

负责 Web 服务和前端展示。
"""
from web.http_server import create_app
from web.sse_handler import SSEHandler

__all__ = [
    "create_app",
    "SSEHandler",
]
