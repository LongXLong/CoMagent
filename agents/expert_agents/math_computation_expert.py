# -*- coding: utf-8 -*-
"""数学计算专家。"""

from agents.expert_agents.base import BaseExpertAgent


class MathComputationExpert(BaseExpertAgent):
	agent_name = "math_computation_expert"
	question_type = "MATH_COMPUTATION"
	expert_title = "数学计算专家（Math Computation Expert）"
	review_items = (
		"计算步骤是否正确、完整，单位与符号是否一致？",
		"是否存在算术错误、公式误用或中间推导跳步？",
		"请给出更严谨的计算与验算建议。",
	)
	default_comment = "可从步骤严谨性与验算机制角度提高数学解答可靠性。"
	default_score = 0.82

