# -*- coding: utf-8 -*-
"""一致性检查通用 Agent。"""

from typing import Any

from agents.general_agents.base import (
    AGENT_OUTPUT_JSON_SCHEMA,
    BaseGeneralAgent,
    parse_agent_output,
)
from utils.llm import llm_call


class ConsistencyChecker(BaseGeneralAgent):
    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = """
你是一致性检查员。请检查回答内部术语、立场、结论是否前后一致，避免自相矛盾。

问题：
{question}

回答：
{answer}

{schema}
""".strip().format(question=question, answer=answer, schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment, score = parse_agent_output(raw, 0.76)
        return {"agent": "consistency_checker", "comment": comment or "请统一术语并避免前后冲突。", "score": score}
