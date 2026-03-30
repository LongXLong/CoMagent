# -*- coding: utf-8 -*-
"""无害性审查通用 Agent。"""

from typing import Any

from agents.general_agents.base import (
    AGENT_OUTPUT_JSON_SCHEMA,
    BaseGeneralAgent,
    parse_agent_output,
)
from utils.llm import llm_call


class HarmlessnessChecker(BaseGeneralAgent):
    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = """
你是无害性审查员。请检查回答是否包含暴力、色情、违法、自残、仇恨、危险操作等有害内容，并给出安全化修改建议。

问题：
{question}

回答：
{answer}

{schema}
""".strip().format(question=question, answer=answer, schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment, score = parse_agent_output(raw, 0.82)
        return {"agent": "harmlessness_checker", "comment": comment or "请移除潜在有害内容，保持输出安全、克制且不鼓励危险行为。", "score": score}
