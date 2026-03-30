# -*- coding: utf-8 -*-
"""流畅自然性审查通用 Agent。"""

from typing import Any

from agents.general_agents.base import (
    AGENT_OUTPUT_JSON_SCHEMA,
    BaseGeneralAgent,
    parse_agent_output,
)
from utils.llm import llm_call


class FluencyEditor(BaseGeneralAgent):
    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = """
你是语言流畅性编辑。请检查回答是否语句通顺、自然、无明显语病、不生硬，并给出简洁可执行的润色建议。

问题：
{question}

回答：
{answer}

{schema}
""".strip().format(question=question, answer=answer, schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment, score = parse_agent_output(raw, 0.77)
        return {"agent": "fluency_editor", "comment": comment or "请优化语句衔接与自然度，避免生硬或不通顺的表达。", "score": score}
