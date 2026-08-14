"""
制度抽象基类 + 注册表.

所有制度实现 BaseRegulation 接口。
RuleRegistry 自动发现 rules/ 子目录中的制度。
"""
import os
import importlib
import pkgutil
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from core.event_bus import EventStream

logger = logging.getLogger("rules.base")


class BaseRule(ABC):
    """所有制度的抽象基类."""

    @abstractmethod
    def name(self) -> str:
        """制度名称，如 'supervision', 'self_ticket'."""
        pass

    @abstractmethod
    def subscribe_events(self, event_bus: EventStream) -> None:
        """声明本制度关心哪些事件."""
        pass

    def is_active(self) -> bool:
        """当前是否有活跃流程."""
        return getattr(self, "_active", False)

    @abstractmethod
    def get_current_flow(self) -> Optional[dict]:
        """获取当前活跃流程."""
        pass

    def finalize(self) -> Optional[dict]:
        """视频结束时关闭流程."""
        if not self.is_active():
            return None
        # 动态分派给子类的 _close_flow
        return self._close_flow(ts=0, source="finalize")

    def _next_flow_id(self) -> int:
        """获取下一个流程 ID."""
        self._flow_counter = getattr(self, "_flow_counter", 0) + 1
        return self._flow_counter

    def reset(self) -> None:
        """新一轮推理前重置制度状态；子类可覆盖以清理更多内部状态."""
        self.finalize()
        self._flow_counter = 0

    def save_results(self, result_dir: str) -> None:
        """保存规则事件到JSON(子类可覆盖)."""
        pass


class RuleRegistry:
    """制度注册表 — 自动发现并管理所有制度."""

    def __init__(self):
        """初始化."""
        self._rules: Dict[str, BaseRule] = {}

    def register(self, rule: BaseRule) -> None:
        """注册."""
        self._rules[rule.name()] = rule
        logger.info(f"注册制度: {rule.name()}")

    def discover(self) -> None:
        """扫描 rules/ 子目录，动态加载规则."""
        try:
            package = importlib.import_module("rules")
            package_path = os.path.dirname(package.__file__)

            for _, name, is_pkg in pkgutil.iter_modules([package_path]):
                if name in ("base", ):
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
        """获取."""
        return self._rules.get(name)

    def all(self) -> List[BaseRule]:
        """全部."""
        return list(self._rules.values())

    def subscribe_all(self, event_bus: EventStream) -> None:
        """订阅全部."""
        for reg in self._rules.values():
            reg.subscribe_events(event_bus)
            logger.info(f"制度 {reg.name()} 已订阅事件")

    def save_all_results(self, result_dir: str) -> None:
        """保存全部results."""
        # 流水线收尾：finalize 关闭活跃流程（触发 FLOW_ENDED）+ 各制度持久化自身产物
        for reg in self._rules.values():
            try:
                flow = reg.finalize()
                if flow:
                    logger.info(
                        f"制度 {reg.name()} finalize 关闭流程 flow_id={flow.get('flow_id')}"
                    )
            except Exception as e:
                logger.error(f"制度 {reg.name()} finalize 失败: {e}",
                             exc_info=True)
            try:
                reg.save_results(result_dir)
            except Exception as e:
                logger.error(f"制度 {reg.name()} save_results 失败: {e}",
                             exc_info=True)
