# -*- coding: utf-8 -*-
"""通用问题专家。"""

from agents.expert_agents.base import BaseExpertAgent


class OtherGeneralQExpert(BaseExpertAgent):
	agent_name = "other_general_q_expert"
	question_type = "OTHER_GENERAL_Q"
	expert_title = "通用问题专家（General Domain Expert）"
	review_items = (
		"回答是否直接回应问题并保持事实一致？",
		"是否存在不必要扩展、含混措辞或潜在误导？",
		"请给出更准确、简洁、可操作的优化建议。",
	)
	default_comment = "可从准确性、清晰性与简洁性角度进一步优化通用回答。"
	default_score = 0.75

