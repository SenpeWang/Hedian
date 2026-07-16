"""
事件对齐聚合器 — 兼容包装层（已拆分为两个独立脚本以进行解耦）
"""
import logging
from typing import Any, Callable, Dict, Optional, Set
from core.inference_bus import InferenceBus
from core.module_sync import ModuleSync

logger = logging.getLogger("core.display_buffer")


class DisplayBuffer:
    """
    事件聚合器包装器，根据 writer_only 动态选择实例化 InferenceBus 或 ModuleSync
    """

    def __init__(
        self,
        fps: float = 30.0,
        immediate_types: Optional[Set[str]] = None,
        expected_modules: Optional[Set[str]] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        writer_only: bool = False,
    ):
        self._writer_only = writer_only
        if writer_only:
            self._impl = InferenceBus(
                fps=fps,
                redis_host=redis_host,
                redis_port=redis_port,
                redis_db=redis_db
            )
        else:
            self._impl = ModuleSync(
                fps=fps,
                immediate_types=immediate_types,
                expected_modules=expected_modules,
                redis_host=redis_host,
                redis_port=redis_port,
                redis_db=redis_db
            )

    @property
    def fps(self) -> float:
        return self._impl.fps

    @fps.setter
    def fps(self, value: float) -> None:
        self._impl.fps = value

    @property
    def _module_timeout(self) -> float:
        return getattr(self._impl, "_module_timeout", 60.0)

    @_module_timeout.setter
    def _module_timeout(self, value: float) -> None:
        if hasattr(self._impl, "_module_timeout"):
            self._impl._module_timeout = value

    def update_module_time(self, module_name: str, sec: float) -> None:
        self._impl.update_module_time(module_name, sec)

    def update_module_snapshot(self, module_name: str, snapshot: Dict[str, Any]) -> None:
        self._impl.update_module_snapshot(module_name, snapshot)

    def remove_module(self, module_name: str) -> None:
        self._impl.remove_module(module_name)

    def set_push_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        if hasattr(self._impl, "set_push_callback"):
            self._impl.set_push_callback(callback)

    def push_display(self, event_type: str, data: Dict[str, Any]) -> None:
        self._impl.push_display(event_type, data)

    def push_sentinel(self) -> None:
        self._impl.push_sentinel()

    def start(self) -> None:
        self._impl.start()

    def stop(self) -> None:
        self._impl.stop()

    def get_stats(self) -> Dict[str, Any]:
        return self._impl.get_stats()

    def clear(self) -> None:
        self._impl.clear()
