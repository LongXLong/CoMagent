# -*- coding: utf-8 -*-
"""创意策划专家。"""

from agents.expert_agents.base import BaseExpertAgent


class CreativeIdeationExpert(BaseExpertAgent):
	agent_name = "creative_ideation_expert"
	question_type = "CREATIVE_IDEATION"
	expert_title = "创意策划专家（Creative Ideation Expert）"
	review_items = (
		"创意是否有新颖性并与目标场景契合？",
		"方案是否具备可执行路径与落地步骤？",
		"请给出提升创意质量与可实施性的建议。",
	)
	default_comment = "可从创意新颖度与落地可行性进一步增强方案。"
	default_score = 0.76

