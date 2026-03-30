# -*- coding: utf-8 -*-
"""文本写作专家。"""

from agents.expert_agents.base import BaseExpertAgent


class TextWritingExpert(BaseExpertAgent):
	agent_name = "text_writing_expert"
	question_type = "TEXT_WRITING"
	expert_title = "文本写作专家（Text Writing Expert）"
	review_items = (
		"结构是否清晰，段落组织是否符合写作目标？",
		"表达是否准确、连贯，是否存在歧义或冗余？",
		"语气、风格与受众是否匹配，并给出可执行改写建议。",
	)
	default_comment = "可从结构、连贯性与表达准确性进一步优化文本写作质量。"
	default_score = 0.78

