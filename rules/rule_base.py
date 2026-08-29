"""制度抽象基类 + 注册表.

所有制度实现 BaseRule 接口。
RuleRegistry 自动发现 rules/ 子目录中的制度。
"""
import importlib
import logging
import os
import pkgutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.event_bus import EventBus

logger = logging.getLogger("rules.base")


class BaseRule(ABC):
    """所有制度的抽象基类."""

    @abstractmethod
    def name(self) -> str:
        """制度名称，如 'supervision', 'self_ticket'."""
        pass

    @abstractmethod
    def subscribe_events(self, event_bus: EventBus) -> None:
        """声明本制度关心哪些事件."""
        pass

    def is_active(self) -> bool:
        """当前是否有活跃流程."""
        return getattr(self, "_active", False)

    def finalize(self) -> Optional[Dict[str, Any]]:
        """视频结束时关闭流程.

        若存在活跃流程则分派给子类的 _close_flow 关闭；否则直接返回 None.

        Returns:
            关闭的流程事件字典；无活跃流程时返回 None.
        """
        if not self.is_active():
            return None
        # 动态分派给子类的 _close_flow
        return self._close_flow(timestamp=0, source="finalize")

    def _next_flow_id(self) -> int:
        """获取下一个流程 ID.

        Returns:
            自增后的流程 ID.
        """
        self._flow_counter = getattr(self, "_flow_counter", 0) + 1
        return self._flow_counter

    def reset(self) -> None:
        """新一轮推理前重置制度状态；子类可覆盖以清理更多内部状态."""
        self.finalize()
        self._flow_counter = 0

    def save_results(self, result_dir: str) -> None:
        """保存规则事件到 JSON（子类可覆盖）.

        Args:
            result_dir: 结果目录路径.
        """
        pass


class RuleRegistry:
    """制度注册表 — 自动发现并管理所有制度."""

    def __init__(self) -> None:
        """初始化制度注册表."""
        self._rules: Dict[str, BaseRule] = {}

    def register(self, rule: BaseRule) -> None:
        """注册制度到注册表.

        Args:
            rule: 制度实例，以 name() 为键.
        """
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
                        rule = module.register()
                        self._rules[rule.name()] = rule
                        logger.info(f"发现制度: {rule.name()}")
                except Exception as error:
                    logger.error(f"加载制度 {name} 失败: {error}", exc_info=True)

        except Exception as error:
            logger.error(f"扫描制度目录失败: {error}", exc_info=True)

    def get_rule(self, name: str) -> Optional[BaseRule]:
        """按名称获取制度实例.

        Args:
            name: 制度名称.

        Returns:
            匹配的制度实例；未注册时返回 None.
        """
        return self._rules.get(name)

    def get_all_rules(self) -> List[BaseRule]:
        """获取全部已注册的制度实例.

        Returns:
            全部制度实例列表.
        """
        return list(self._rules.values())

    def save_all_results(self, result_dir: str) -> None:
        """对全部制度执行收尾 finalize 并保存各自产物.

        Args:
            result_dir: 结果目录路径.
        """
        # 流水线收尾：finalize 关闭活跃流程（触发 FLOW_ENDED）+ 各制度持久化自身产物
        for rule in self._rules.values():
            try:
                flow = rule.finalize()
                if flow:
                    logger.info(
                        f"制度 {rule.name()} finalize 关闭流程 flow_id={flow.get('flow_id')}"
                    )
            except Exception as error:
                logger.error(f"制度 {rule.name()} finalize 失败: {error}",
                             exc_info=True)
            try:
                rule.save_results(result_dir)
            except Exception as error:
                logger.error(f"制度 {rule.name()} save_results 失败: {error}",
                             exc_info=True)
