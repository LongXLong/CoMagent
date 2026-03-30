# -*- coding: utf-8 -*-
"""营销文案专家。"""

from agents.expert_agents.base import BaseExpertAgent


class MarketingCopywritingExpert(BaseExpertAgent):
    agent_name = "marketing_copywriting_expert"
    question_type = "MARKETING_COPYWRITING"
    expert_title = "营销文案专家（Marketing Copywriting Expert）"
    review_items = (
        "文案是否聚焦卖点、受众痛点与行动号召？",
        "信息层次是否清晰，是否具备传播与转化潜力？",
        "请给出提升吸引力与转化率的改进建议。",
    )
    default_comment = "可从卖点表达、受众触达与转化引导角度优化文案。"
    default_score = 0.79
