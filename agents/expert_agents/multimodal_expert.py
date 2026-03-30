# -*- coding: utf-8 -*-
"""多模态专家。"""

from agents.expert_agents.base import BaseExpertAgent


class MultimodalExpert(BaseExpertAgent):
    agent_name = "multimodal_expert"
    question_type = "MULTIMODAL"
    expert_title = "多模态专家（Multimodal Expert）"
    review_items = (
        "是否正确整合文本、图像/音频等多模态信息？",
        "模态间信息是否一致，是否出现跨模态冲突？",
        "请给出提升多模态融合质量与可解释性的建议。",
    )
    default_comment = "可从模态对齐与融合一致性角度优化多模态回答。"
    default_score = 0.77
