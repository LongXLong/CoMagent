# -*- coding: utf-8 -*-
"""完整性检查通用 Agent。"""

from typing import Any

from agents.general_agents.base import (
    AGENT_OUTPUT_JSON_SCHEMA,
    BaseGeneralAgent,
    parse_agent_output,
)
from utils.llm import llm_call


class CompletenessChecker(BaseGeneralAgent):
    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = """
你是完整性检查员。请判断回答是否完整覆盖问题要求，是否遗漏边界条件、步骤或结论。

问题：
{question}

回答：
{answer}

{schema}
""".strip().format(question=question, answer=answer, schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment, score = parse_agent_output(raw, 0.78)
        return {"agent": "completeness_checker", "comment": comment or "请补全遗漏点并明确结论。", "score": score}
