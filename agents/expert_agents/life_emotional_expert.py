# -*- coding: utf-8 -*-
"""生活情感专家。"""

from agents.expert_agents.base import BaseExpertAgent


class LifeEmotionalExpert(BaseExpertAgent):
	agent_name = "life_emotional_expert"
	question_type = "LIFE_EMOTIONAL"
	expert_title = "生活情感专家（Life & Emotional Expert）"
	review_items = (
		"建议是否体现共情、尊重与现实可行性？",
		"是否避免绝对化判断，并考虑情绪与边界管理？",
		"请给出更温和、可执行且支持性的建议。",
	)
	default_comment = "可从共情表达与可执行支持建议角度进一步优化。"
	default_score = 0.76

