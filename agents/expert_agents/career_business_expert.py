# -*- coding: utf-8 -*-
"""职业与商业专家。"""

from agents.expert_agents.base import BaseExpertAgent


class CareerBusinessExpert(BaseExpertAgent):
    agent_name = "career_business_expert"
    question_type = "CAREER_BUSINESS"
    expert_title = "职业与商业专家（Career & Business Expert）"
    review_items = (
        "建议是否符合职业发展与商业场景的现实约束？",
        "是否考虑资源、风险、收益与执行路径？",
        "请给出更可落地的职业/商业改进建议。",
    )
    default_comment = "可从战略可行性与执行路径角度增强职业商业建议。"
    default_score = 0.78
