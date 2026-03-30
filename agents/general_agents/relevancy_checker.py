# -*- coding: utf-8 -*-
"""相关性审查通用 Agent。"""

from typing import Any

from agents.general_agents.base import (
    AGENT_OUTPUT_JSON_SCHEMA,
    BaseGeneralAgent,
    parse_agent_output,
)
from utils.llm import llm_call


class RelevancyChecker(BaseGeneralAgent):
    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = """
你是相关性审查员。请检查回答是否严格贴合问题，是否跑题、夹带无关内容、废话过多，并给出精简而直接的修正建议。

问题：
{question}

回答：
{answer}

{schema}
""".strip().format(question=question, answer=answer, schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment, score = parse_agent_output(raw, 0.78)
        return {"agent": "relevancy_checker", "comment": comment or "请更严格围绕问题作答，删除跑题或冗余内容。", "score": score}
