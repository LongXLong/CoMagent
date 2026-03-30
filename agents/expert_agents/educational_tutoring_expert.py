# -*- coding: utf-8 -*-
"""教育辅导专家。"""

from agents.expert_agents.base import BaseExpertAgent


class EducationalTutoringExpert(BaseExpertAgent):
	agent_name = "educational_tutoring_expert"
	question_type = "EDUCATIONAL_TUTORING"
	expert_title = "教育辅导专家（Educational Tutoring Expert）"
	review_items = (
		"讲解是否循序渐进、符合学习者认知水平？",
		"是否包含关键概念、示例与纠错提示？",
		"请给出可执行的教学改进建议，提高可学性。",
	)
	default_comment = "可从教学分层、例题与反馈机制角度优化辅导质量。"
	default_score = 0.78

