# -*- coding: utf-8 -*-
"""证据与可验证性通用 Agent。"""

from typing import Any

from agents.general_agents.base import (
    AGENT_OUTPUT_JSON_SCHEMA,
    BaseGeneralAgent,
    parse_agent_output,
)
from utils.llm import llm_call


class EvidenceChecker(BaseGeneralAgent):
    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = """
你是证据与可验证性审查员。请识别回答中缺证据、过度断言、不可验证表述，并给出修正建议。

问题：
{question}

回答：
{answer}

{schema}
""".strip().format(question=question, answer=answer, schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment, score = parse_agent_output(raw, 0.8)
        return {"agent": "evidence_checker", "comment": comment or "请补充可验证依据并降低过度断言。", "score": score}
