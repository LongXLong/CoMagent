# -*- coding: utf-8 -*-
"""逻辑审查通用 Agent。"""

from typing import Any

from agents.general_agents.base import (
    AGENT_OUTPUT_JSON_SCHEMA,
    BaseGeneralAgent,
    parse_agent_output,
)
from utils.llm import llm_call


class LogicChecker(BaseGeneralAgent):
    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = """
你是逻辑审校员。请检查回答是否存在逻辑跳跃、因果倒置、前后矛盾，并给出可执行修改建议。

问题：
{question}

回答：
{answer}

{schema}
""".strip().format(question=question, answer=answer, schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment, score = parse_agent_output(raw, 0.8)
        return {"agent": "logic_checker", "comment": comment or "请增强逻辑链条与因果一致性。", "score": score}
