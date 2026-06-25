"""
制度层

包含所有制度：
- supervision_regulation: 监护制度
- self_ticket_regulation: 自唱票制度

所有制度继承 BaseRegulation，实现统一接口。
"""
from regulations.regulation_base import BaseRegulation, RegulationRegistry
from regulations.supervision_regulation import SupervisionRegulation
from regulations.self_ticket_regulation import SelfTicketRegulation

__all__ = [
    "BaseRegulation",
    "RegulationRegistry",
    "SupervisionRegulation",
    "SelfTicketRegulation",
]
