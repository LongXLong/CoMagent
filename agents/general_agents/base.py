# -*- coding: utf-8 -*-
"""通用 Agent 基类与公共解析逻辑。"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any

AGENT_OUTPUT_JSON_SCHEMA = """
请务必只输出一个合法的 JSON 对象，不要输出任何其他文字、解释或 Markdown 标记。
JSON 格式固定为（仅此一个字段）：
{
    "comment": "简短评论内容，指出问题与改进建议（1-3 句）"
}
不要输出 ```json 等代码块标记，只输出纯 JSON。
"""


def parse_agent_output(raw: str, default_score: float) -> tuple[str, float]:
    """从 LLM 回复中解析出 comment 与 score。"""
    if not (raw and raw.strip()):
        return "", default_score
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        obj = json.loads(raw)
        comment = (obj.get("comment") or "").strip()
        return comment or "", default_score
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(raw[start : i + 1])
                        comment = (obj.get("comment") or "").strip()
                        return comment or "", default_score
                    except (json.JSONDecodeError, TypeError, ValueError):
                        break
    return raw, default_score


class BaseGeneralAgent(ABC):
    """通用 Agent 基类。"""

    @abstractmethod
    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        """对回答进行质量审查。"""
        ...
