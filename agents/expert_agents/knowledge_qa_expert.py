# -*- coding: utf-8 -*-
"""知识问答专家。"""

from agents.expert_agents.base import BaseExpertAgent


class KnowledgeQAExpert(BaseExpertAgent):
	agent_name = "knowledge_qa_expert"
	question_type = "KNOWLEDGE_QA"
	expert_title = "知识问答专家（Knowledge QA Expert）"
	review_items = (
		"事实性回答是否准确且与问题直接对应？",
		"是否存在时间、人物、地点、数字等关键事实错误？",
		"请给出更可验证、更直接的答案改进建议。",
	)
	default_comment = "可从事实核验与直答性角度进一步提升问答质量。"
	default_score = 0.8

