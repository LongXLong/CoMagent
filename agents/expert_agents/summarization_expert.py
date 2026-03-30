# -*- coding: utf-8 -*-
"""摘要总结专家。"""

from agents.expert_agents.base import BaseExpertAgent


class SummarizationExpert(BaseExpertAgent):
	agent_name = "summarization_expert"
	question_type = "SUMMARIZATION"
	expert_title = "摘要总结专家（Summarization Expert）"
	review_items = (
		"摘要是否覆盖原信息的核心要点且无关键遗漏？",
		"摘要是否避免引入原文不存在的信息与主观扩展？",
		"信息压缩比例是否合理，并给出更精炼的改进建议。",
	)
	default_comment = "可进一步提升摘要完整性、忠实度与压缩效率。"
	default_score = 0.78

