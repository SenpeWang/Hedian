"""
制度抽象基类 + 注册表

所有制度实现 BaseRegulation 接口。
RuleRegistry 自动发现 rules/ 子目录中的制度。
"""
import os
import importlib
import pkgutil
import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from core.event_bus import EventBus

logger = logging.getLogger("rules.base")


class BaseRule(ABC):
    """所有制度的抽象基类"""

    @abstractmethod
    def name(self) -> str:
        """制度名称，如 'supervision', 'self_ticket'"""
        pass

    @abstractmethod
    def subscribe_events(self, event_bus: EventBus) -> None:
        """声明本制度关心哪些事件"""
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """当前是否有活跃流程"""
        pass

    @abstractmethod
    def get_current_flow(self) -> Optional[dict]:
        """获取当前活跃流程"""
        pass

    @abstractmethod
    def finalize(self) -> Optional[dict]:
        """视频结束时关闭流程"""
        pass


class RuleRegistry:
    """制度注册表 — 自动发现并管理所有制度"""

    def __init__(self):
        self._rules: Dict[str, BaseRule] = {}

    def register(self, rule: BaseRule) -> None:
        """
        手动注册一个制度

        Args:
            regulation: 制度实例
        """
        self._rules[rule.name()] = rule
        logger.info(f"注册制度: {regulation.name()}")

    def discover(self) -> None:
        """扫描 rules/ 子目录，动态加载规则"""
        try:
            package = importlib.import_module("rules")
            package_path = os.path.dirname(package.__file__)

            for _, name, is_pkg in pkgutil.iter_modules([package_path]):
                if name in ("base",):
                    continue
                try:
                    module = importlib.import_module(f"rules.{name}")
                    if hasattr(module, "register"):
                        reg = module.register()
                        self._rules[reg.name()] = reg
                        logger.info(f"发现制度: {reg.name()}")
                except Exception as e:
                    logger.error(f"加载制度 {name} 失败: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"扫描制度目录失败: {e}", exc_info=True)

    def get(self, name: str) -> Optional[BaseRule]:
        """
        获取制度

        Args:
            name: 制度名称

        Returns:
            制度实例，或 None
        """
        return self._rules.get(name)

    def all(self) -> List[BaseRule]:
        """
        获取所有制度

        Returns:
            制度列表
        """
        return list(self._rules.values())

    def subscribe_all(self, event_bus: EventBus) -> None:
        """
        让所有已注册制度订阅 event_bus 事件

        Args:
            event_bus: 消息总线
        """
        for reg in self._rules.values():
            reg.subscribe_events(event_bus)
            logger.info(f"制度 {reg.name()} 已订阅事件")
