# -*- coding: utf-8 -*-
"""合规性审查通用 Agent。"""

from typing import Any

from agents.general_agents.base import (
    AGENT_OUTPUT_JSON_SCHEMA,
    BaseGeneralAgent,
    parse_agent_output,
)
from utils.llm import llm_call


class ComplianceChecker(BaseGeneralAgent):
    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = """
你是合规性审查员。请检查回答是否违反平台政策、隐私保护要求、数据安全要求，是否存在敏感信息泄露或不当指导，并给出合规修改建议。

问题：
{question}

回答：
{answer}

{schema}
""".strip().format(question=question, answer=answer, schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment, score = parse_agent_output(raw, 0.82)
        return {"agent": "compliance_checker", "comment": comment or "请避免违反政策或泄露隐私信息，确保输出符合合规与安全要求。", "score": score}
