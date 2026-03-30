# -*- coding: utf-8 -*-
"""通用 Agent 工厂。"""

import random

from agents.general_agents.base import BaseGeneralAgent
from agents.general_agents.brevity_advisor import BrevityAdvisor
from agents.general_agents.clarity_editor import ClarityEditor
from agents.general_agents.compliance_checker import ComplianceChecker
from agents.general_agents.completeness_checker import CompletenessChecker
from agents.general_agents.consistency_checker import ConsistencyChecker
from agents.general_agents.evidence_checker import EvidenceChecker
from agents.general_agents.fluency_editor import FluencyEditor
from agents.general_agents.harmlessness_checker import HarmlessnessChecker
from agents.general_agents.logic_checker import LogicChecker
from agents.general_agents.relevancy_checker import RelevancyChecker


class GeneralAgentFactory:
    agents: dict[str, BaseGeneralAgent] = {
        "logic_checker": LogicChecker(),
        "clarity_editor": ClarityEditor(),
        "completeness_checker": CompletenessChecker(),
        "evidence_checker": EvidenceChecker(),
        "brevity_advisor": BrevityAdvisor(),
        "consistency_checker": ConsistencyChecker(),
        "relevancy_checker": RelevancyChecker(),
        "harmlessness_checker": HarmlessnessChecker(),
        "compliance_checker": ComplianceChecker(),
        "fluency_editor": FluencyEditor(),
    }

    @staticmethod
    def all_names() -> list[str]:
        return list(GeneralAgentFactory.agents.keys())

    @staticmethod
    def sample_names(k: int = 3) -> list[str]:
        names = GeneralAgentFactory.all_names()
        if len(names) <= k:
            return names
        return random.sample(names, k)

    @staticmethod
    def create(agent_name: str) -> BaseGeneralAgent:
        return GeneralAgentFactory.agents.get(agent_name, LogicChecker())
