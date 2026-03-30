# -*- coding: utf-8 -*-
"""Meta-Knowledge (MK)：策略控制中心，从 MK 数据按问题类型选择 Agent、控制循环、决策是否继续。"""

from typing import Any

from agents.expert_agents.factory import ExpertAgentFactory
from agents.general_agents.factory import GeneralAgentFactory
from config import OPENAI_MODEL
from memory import MAB_ALGORITHM, get_config_for_question_type, infer_question_type, load_mk, select_agents_by_mab
from utils.llm import llm_call, semantic_similarity
from utils.logger import get_logger

logger = get_logger(__name__)

# 全部可选通用 Agent 名称（专家 Agent 按 question_type 唯一映射）
ALL_AGENT_NAMES = GeneralAgentFactory.all_names()
GENERAL_AGENT_WHITELISTS: dict[str, list[str]] = {
    "TEXT_WRITING": [
        "clarity_editor",
        "fluency_editor",
        "relevancy_checker",
        "brevity_advisor",
    ],
    "SUMMARIZATION": [
        "relevancy_checker",
        "completeness_checker",
        "brevity_advisor",
        "clarity_editor",
    ],
    "CODE_DEVELOPMENT": [
        "logic_checker",
        "completeness_checker",
        "consistency_checker",
        "relevancy_checker",
    ],
    "KNOWLEDGE_QA": [
        "evidence_checker",
        "clarity_editor",
        "consistency_checker",
        "relevancy_checker",
    ],
    "EDUCATIONAL_TUTORING": [
        "clarity_editor",
        "completeness_checker",
        "logic_checker",
        "fluency_editor",
    ],
    "TRANSLATION_LOCALIZATION": [
        "fluency_editor",
        "consistency_checker",
        "relevancy_checker",
        "clarity_editor",
    ],
    "CREATIVE_IDEATION": [
        "relevancy_checker",
        "clarity_editor",
        "fluency_editor",
        "brevity_advisor",
    ],
    "DATA_PROCESSING": [
        "completeness_checker",
        "logic_checker",
        "consistency_checker",
        "relevancy_checker",
    ],
    "ROLE_PLAYING": [
        "relevancy_checker",
        "consistency_checker",
        "fluency_editor",
        "harmlessness_checker",
    ],
    "CAREER_BUSINESS": [
        "relevancy_checker",
        "clarity_editor",
        "compliance_checker",
        "evidence_checker",
    ],
    "LIFE_EMOTIONAL": [
        "harmlessness_checker",
        "compliance_checker",
        "fluency_editor",
        "relevancy_checker",
    ],
    "MARKETING_COPYWRITING": [
        "relevancy_checker",
        "fluency_editor",
        "clarity_editor",
        "compliance_checker",
    ],
    "LOGICAL_REASONING": [
        "logic_checker",
        "consistency_checker",
        "completeness_checker",
        "relevancy_checker",
    ],
    "MATH_COMPUTATION": [
        "logic_checker",
        "completeness_checker",
        "consistency_checker",
        "relevancy_checker",
    ],
    "MULTIMODAL": [
        "relevancy_checker",
        "clarity_editor",
        "completeness_checker",
        "harmlessness_checker",
    ],
    "OTHER_GENERAL_Q": [
        "relevancy_checker",
        "clarity_editor",
        "consistency_checker",
        "evidence_checker",
    ],
}


class MetaKnowledge:
    """策略中枢：从 MK 数据中按问题类型读取策略、优先级、阈值，选择参与反馈的 Agent、决策是否继续优化。"""

    def __init__(self, mk: dict[str, Any] | None = None):
        self.mk = mk if mk is not None else load_mk()
        self._current_config: dict[str, Any] = {}
        self._current_question_type: str = ""
        self._last_general_agents: list[str] = []

    def _ensure_config(self, question: str | None = None, question_type: str | None = None) -> None:
        """
        加载当前问题类型对应的 MK 配置，供 select_agents / should_continue 使用。
        若传入 question_type 则直接按类型取配置；否则根据 question 推断类型。
        """
        if question_type is not None:
            self._current_question_type = question_type
        elif question is not None:
            self._current_question_type = infer_question_type(question, self.mk)
        else:
            return
        self._current_config = get_config_for_question_type(self.mk, self._current_question_type)

    def _get_candidate_general_agents(self) -> list[str]:
        candidates = GENERAL_AGENT_WHITELISTS.get(self._current_question_type, ALL_AGENT_NAMES)
        valid_names = set(ALL_AGENT_NAMES)
        filtered = [name for name in candidates if name in valid_names]
        return filtered or ALL_AGENT_NAMES

    def select_agents(
        self,
        question: str,
        confidence: float,
        consistency: float,
        question_type: str | None = None,
    ) -> list[str]:
        """
        Use the configured MAB algorithm to select 3 general agents, then
        add 1 question-type expert agent.
        """
        self._ensure_config(question=question, question_type=question_type)
        candidate_general_agents = self._get_candidate_general_agents()
        general_three, mab_scores = select_agents_by_mab(
            candidate_general_agents,
            self._current_config.get("agent_mab_stats", {}),
            k=3,
            algo=MAB_ALGORITHM,
        )
        expert = ExpertAgentFactory.create_by_question_type(self._current_question_type)
        expert_name = expert.agent_name
        self._last_general_agents = list(general_three)
        selected = general_three + [expert_name]
        logger.info(
            "MK 选 Agent: %s (question_type=%s, expert=%s, mab=%s, candidate_generals=%s, scores=%s)",
            selected,
            self._current_question_type,
            expert_name,
            MAB_ALGORITHM,
            candidate_general_agents,
            mab_scores,
        )
        return selected

    def get_last_general_agents(self) -> list[str]:
        """Return the last round of general agents selected by MAB."""
        return list(self._last_general_agents)

    def should_continue(
        self,
        prev_answer: str,
        new_answer: str,
        improvement_score: float,
        loop_count: int,
        *,
        initial_answer: str | None = None,
    ) -> bool:
        """
        判定是否继续下一轮反思循环。
        约定：该函数在每一轮结束后调用，loop_count 为当前轮结束前的计数（首轮后为 0）。
        规则：
        - 若双向蕴含通过（语义基本等价），停止；
        - 若达到最大轮数，停止；
        - 若相对上一轮改动过小（高相似度）或 improvement_score 低于 improvement_min，停止；
        - 若相对首答无继续变好趋势，停止；
        - 否则继续下一轮。
        """
        
        if not self._current_config:
            return False
        strategy = self._current_config.get("strategy", {})
        max_loops = int(strategy.get("max_loops", 3))
        similarity_threshold = float(strategy.get("similarity_threshold", 0.9))
        improvement_min = float(strategy.get("improvement_min", 0.05))

        if self.is_bientail(prev_answer, new_answer):
            logger.info("should_continue 结束原因: 双向蕴含 (bidirectional entailment) 检测PASS，停止循环")
            return False

        # 当前轮结束后累计轮数
        next_loop_count = loop_count + 1
        if next_loop_count >= max_loops:
            logger.info("should_continue 结束原因: 达到设置的最高轮数 max_loops=%s (当前 loop_count=%s)", max_loops, loop_count)
            return False

        # 与上一轮回答的一致性：复用外部已算好的 improvement_score，避免重复算 embedding
        sim_new_prev = max(0.0, min(1.0, 1.0 - improvement_score))
        if sim_new_prev > similarity_threshold:
            logger.info(
                "should_continue 结束原因: 改动过小 (与上一轮相似度 %.3f > 阈值 %.3f，设置的最高轮数 max_loops=%s)",
                sim_new_prev, similarity_threshold, max_loops,
            )
            return False

        if improvement_score < improvement_min:
            logger.info(
                "should_continue 结束原因: 改进不足 (improvement_score=%.3f < improvement_min=%.3f，max_loops=%s)",
                improvement_score, improvement_min, max_loops,
            )
            return False

        # 与首答的一致性趋势：从第 2 轮结束后开始判断，避免第 1 轮时 prev==initial 导致必停
        if loop_count >= 1 and initial_answer and initial_answer.strip():
            sim_new_initial = semantic_similarity(new_answer, initial_answer)
            sim_prev_initial = semantic_similarity(prev_answer, initial_answer)
            if sim_new_initial <= sim_prev_initial:
                logger.info(
                    "should_continue 结束原因: 无变好趋势 (sim_new_initial=%.3f <= sim_prev_initial=%.3f)，设置的最高轮数 max_loops=%s",
                    sim_new_initial, sim_prev_initial, max_loops,
                )
                return False
        logger.info(
            "should_continue 继续: 满足继续条件 (sim_new_prev=%.3f，阈值=%.3f，improvement_score=%.3f，improvement_min=%.3f，max_loops=%s)",
            sim_new_prev, similarity_threshold, improvement_score, improvement_min, max_loops
        )
        return True
    
    def is_bientail(
        self,
        answer1: str,
        answer2: str,
    ) -> bool:
        """
        判定两个回答是否为双向蕴含 (bidirectional entailment)。
        使用 LLM 进行双向蕴含判断：
        1. 判断 answer1 是否语义蕴含 answer2
        2. 判断 answer2 是否语义蕴含 answer1
        若两个方向均为 Yes，则认为两个回答语义等价，返回 True。

        Prompt 参考自 STaR (Self-Taught Reasoner) 论文中关于语义等价性检测的方法。
        """
        # 双向蕴含判断 Prompt 模板
        logger.info("双向蕴含判断开始: answer1='%s', answer2='%s'", answer1, answer2)
        entail_prompt_template = """You are a semantic equivalence judge. Given two answers, determine if the first answer semantically entails the second answer.

Answer 1: {answer1}
Answer 2: {answer2}

Does Answer 1 semantically entail Answer 2? That is, if Answer 1 is true, must Answer 2 also be true?
Reply with "Yes" or "No" only."""

        # 方向1: answer1 -> answer2
        prompt_1_to_2 = entail_prompt_template.format(answer1=answer1, answer2=answer2)
        try:
            response_1_to_2 = llm_call(
                prompt_1_to_2, model=OPENAI_MODEL, temperature=0.0
            ).strip().lower()
        except Exception as e:
            logger.warning("is_bientail 调用 LLM 失败 (1->2): %s", e)
            return False

        # 方向2: answer2 -> answer1
        prompt_2_to_1 = entail_prompt_template.format(answer1=answer2, answer2=answer1)
        try:
            response_2_to_1 = llm_call(
                prompt_2_to_1, model=OPENAI_MODEL, temperature=0.0
            ).strip().lower()
        except Exception as e:
            logger.warning("is_bientail 调用 LLM 失败 (2->1): %s", e)
            return False

        # 双向判断：两个方向都为 Yes 才认为是双向蕴含
        entail_1_to_2 = response_1_to_2.startswith("yes")
        entail_2_to_1 = response_2_to_1.startswith("yes")

        result = entail_1_to_2 and entail_2_to_1
        logger.info(
            "双向蕴含 判断结果: %s (1->2: %s, 2->1: %s)",
            result, response_1_to_2, response_2_to_1
        )
        return result
        
