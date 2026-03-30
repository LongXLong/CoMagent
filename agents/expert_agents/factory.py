# -*- coding: utf-8 -*-
"""专家 Agent 工厂：按名称返回对应领域专家。"""

from .base import BaseExpertAgent
from .career_business_expert import CareerBusinessExpert
from .code_development_expert import CodeDevelopmentExpert
from .creative_ideation_expert import CreativeIdeationExpert
from .data_processing_expert import DataProcessingExpert
from .educational_tutoring_expert import EducationalTutoringExpert
from .knowledge_qa_expert import KnowledgeQAExpert
from .life_emotional_expert import LifeEmotionalExpert
from .logical_reasoning_expert import LogicalReasoningExpert
from .marketing_copywriting_expert import MarketingCopywritingExpert
from .math_computation_expert import MathComputationExpert
from .multimodal_expert import MultimodalExpert
from .other_general_q_expert import OtherGeneralQExpert
from .role_playing_expert import RolePlayingExpert
from .summarization_expert import SummarizationExpert
from .text_writing_expert import TextWritingExpert
from .translation_localization_expert import TranslationLocalizationExpert


class ExpertAgentFactory:
    """专家 Agent 工厂。"""

    experts: dict[str, BaseExpertAgent] = {
        "text_writing_expert": TextWritingExpert(),
        "summarization_expert": SummarizationExpert(),
        "code_development_expert": CodeDevelopmentExpert(),
        "knowledge_qa_expert": KnowledgeQAExpert(),
        "educational_tutoring_expert": EducationalTutoringExpert(),
        "translation_localization_expert": TranslationLocalizationExpert(),
        "creative_ideation_expert": CreativeIdeationExpert(),
        "data_processing_expert": DataProcessingExpert(),
        "role_playing_expert": RolePlayingExpert(),
        "career_business_expert": CareerBusinessExpert(),
        "life_emotional_expert": LifeEmotionalExpert(),
        "marketing_copywriting_expert": MarketingCopywritingExpert(),
        "logical_reasoning_expert": LogicalReasoningExpert(),
        "math_computation_expert": MathComputationExpert(),
        "multimodal_expert": MultimodalExpert(),
        "other_general_q_expert": OtherGeneralQExpert(),
    }
    question_type_to_expert: dict[str, str] = {
        "TEXT_WRITING": "text_writing_expert",
        "SUMMARIZATION": "summarization_expert",
        "CODE_DEVELOPMENT": "code_development_expert",
        "KNOWLEDGE_QA": "knowledge_qa_expert",
        "EDUCATIONAL_TUTORING": "educational_tutoring_expert",
        "TRANSLATION_LOCALIZATION": "translation_localization_expert",
        "CREATIVE_IDEATION": "creative_ideation_expert",
        "DATA_PROCESSING": "data_processing_expert",
        "ROLE_PLAYING": "role_playing_expert",
        "CAREER_BUSINESS": "career_business_expert",
        "LIFE_EMOTIONAL": "life_emotional_expert",
        "MARKETING_COPYWRITING": "marketing_copywriting_expert",
        "LOGICAL_REASONING": "logical_reasoning_expert",
        "MATH_COMPUTATION": "math_computation_expert",
        "MULTIMODAL": "multimodal_expert",
        "OTHER_GENERAL_Q": "other_general_q_expert",
    }

    @staticmethod
    def create(expert_name: str) -> BaseExpertAgent:
        return ExpertAgentFactory.experts.get(expert_name, OtherGeneralQExpert())

    @staticmethod
    def create_by_question_type(question_type: str) -> BaseExpertAgent:
        name = ExpertAgentFactory.question_type_to_expert.get(
            (question_type or "").strip().upper(),
            "other_general_q_expert",
        )
        return ExpertAgentFactory.create(name)
