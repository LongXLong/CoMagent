# -*- coding: utf-8 -*-
"""清晰度优化通用 Agent。"""

from typing import Any

from agents.general_agents.base import (
    AGENT_OUTPUT_JSON_SCHEMA,
    BaseGeneralAgent,
    parse_agent_output,
)
from utils.llm import llm_call


class ClarityEditor(BaseGeneralAgent):
    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = """
你是表达清晰度编辑。请审查回答的结构、可读性与术语解释是否清晰，输出高价值的精简修改建议。

问题：
{question}

回答：
{answer}

{schema}
""".strip().format(question=question, answer=answer, schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment, score = parse_agent_output(raw, 0.75)
        return {"agent": "clarity_editor", "comment": comment or "请增强结构清晰度并减少歧义表述。", "score": score}
