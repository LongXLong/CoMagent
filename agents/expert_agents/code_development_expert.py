# -*- coding: utf-8 -*-
"""代码开发专家。"""

from agents.expert_agents.base import BaseExpertAgent


class CodeDevelopmentExpert(BaseExpertAgent):
	agent_name = "code_development_expert"
	question_type = "CODE_DEVELOPMENT"
	expert_title = "代码开发专家（Code Development Expert）"
	review_items = (
		"技术方案是否正确、可实现，并满足问题约束？",
		"是否存在潜在 bug、边界遗漏或性能/安全风险？",
		"请给出更稳健的实现建议与必要的测试建议。",
	)
	default_comment = "可从正确性、可维护性与测试覆盖角度增强代码方案。"
	default_score = 0.8

