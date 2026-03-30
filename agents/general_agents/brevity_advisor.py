# -*- coding: utf-8 -*-
"""简洁性建议通用 Agent。"""

from typing import Any

from agents.general_agents.base import (
    AGENT_OUTPUT_JSON_SCHEMA,
    BaseGeneralAgent,
    parse_agent_output,
)
from utils.llm import llm_call


class BrevityAdvisor(BaseGeneralAgent):
    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = """
你是简洁性顾问。请在不损失关键信息的前提下指出冗余表达，并给出压缩建议。

问题：
{question}

回答：
{answer}

{schema}
""".strip().format(question=question, answer=answer, schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment, score = parse_agent_output(raw, 0.72)
        return {"agent": "brevity_advisor", "comment": comment or "请压缩冗余句并保留关键信息。", "score": score}
