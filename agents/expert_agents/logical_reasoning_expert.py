# -*- coding: utf-8 -*-
"""逻辑推理专家。"""

from agents.expert_agents.base import BaseExpertAgent


class LogicalReasoningExpert(BaseExpertAgent):
	agent_name = "logical_reasoning_expert"
	question_type = "LOGICAL_REASONING"
	expert_title = "逻辑推理专家（Logical Reasoning Expert）"
	review_items = (
		"推理链是否完整，自前提到结论是否成立？",
		"是否存在逻辑谬误、偷换概念或循环论证？",
		"请给出提升论证严密性的具体改进建议。",
	)
	default_comment = "可从推理链完整性与谬误修复角度增强论证质量。"
	default_score = 0.8

