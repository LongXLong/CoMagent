# -*- coding: utf-8 -*-
"""数据处理专家。"""

from agents.expert_agents.base import BaseExpertAgent


class DataProcessingExpert(BaseExpertAgent):
	agent_name = "data_processing_expert"
	question_type = "DATA_PROCESSING"
	expert_title = "数据处理专家（Data Processing Expert）"
	review_items = (
		"数据清洗、转换、聚合流程是否完整且可复现？",
		"是否考虑缺失值、异常值、偏差与数据质量控制？",
		"请给出更稳健的数据处理优化建议。",
	)
	default_comment = "可从数据质量与处理流程鲁棒性角度优化答案。"
	default_score = 0.8

