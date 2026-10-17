"""制度层.

包含所有制度：
- SupervisionRule: 监护制度
- SelfTicketRule: 自唱票制度
- InfoNoticeRule: 信息通报制度
- NormalCommRule: 普通沟通流程
- PersonnelStatusRule: 人员状态监控制度

所有制度继承 BaseRule，实现统一接口，并由 RuleRegistry 动态发现注册。
"""

from rules.rule_base import BaseRule, RuleRegistry
from rules.supervision_rule import SupervisionRule
from rules.self_ticket_rule import SelfTicketRule
from rules.personnel_status_rule import PersonnelStatusRule
from rules.info_notice_rule import InfoNoticeRule
from rules.normal_comm_rule import NormalCommRule
from rules.flow_recorder import FlowEventRecorder

__all__ = [
    "BaseRule",
    "RuleRegistry",
    "SupervisionRule",
    "SelfTicketRule",
    "PersonnelStatusRule",
    "InfoNoticeRule",
    "NormalCommRule",
    "FlowEventRecorder",
]
