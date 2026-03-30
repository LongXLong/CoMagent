# -*- coding: utf-8 -*-
"""翻译本地化专家。"""

from agents.expert_agents.base import BaseExpertAgent


class TranslationLocalizationExpert(BaseExpertAgent):
    agent_name = "translation_localization_expert"
    question_type = "TRANSLATION_LOCALIZATION"
    expert_title = "翻译本地化专家（Translation & Localization Expert）"
    review_items = (
        "译文是否忠实原意且语义完整，无漏译误译？",
        "是否符合目标语言表达习惯与文化语境？",
        "请给出术语统一与本地化优化建议。",
    )
    default_comment = "可从忠实度、流畅度与本地化适配角度进一步优化。"
    default_score = 0.8
