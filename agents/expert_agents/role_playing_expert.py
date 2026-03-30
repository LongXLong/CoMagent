# -*- coding: utf-8 -*-
"""角色扮演专家。"""

from agents.expert_agents.base import BaseExpertAgent


class RolePlayingExpert(BaseExpertAgent):
	agent_name = "role_playing_expert"
	question_type = "ROLE_PLAYING"
	expert_title = "角色扮演专家（Role Playing Expert）"
	review_items = (
		"角色设定是否一致，语气与行为是否符合人设？",
		"情境推进是否自然，是否出现角色越界或设定冲突？",
		"请给出增强沉浸感与连贯性的改进建议。",
	)
	default_comment = "可从人设一致性与情境连贯性角度提升角色扮演质量。"
	default_score = 0.77

