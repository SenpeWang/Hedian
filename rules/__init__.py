"""
制度层

包含所有制度：
- supervision_regulation: 监护制度
- self_ticket_regulation: 自唱票制度

所有制度继承 BaseRegulation，实现统一接口。
"""
from rules.rule_base import BaseRule, RuleRegistry
from rules.supervision_rule import SupervisionRule
from rules.self_ticket_rule import SelfTicketRule
from rules.personnel_status_rule import PersonnelStatusRule
from rules.info_notice_rule import InfoNoticeRule

__all__ = [
    "BaseRule",
    "RuleRegistry",
    "SupervisionRule",
    "SelfTicketRule",
    "PersonnelStatusRule",
    "InfoNoticeRule",
]
