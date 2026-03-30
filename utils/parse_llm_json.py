# -*- coding: utf-8 -*-
"""
将 LLM 返回的文本解析为字典，供各处统一用 .get() 提取字段。
适用于所有要求 LLM 输出 JSON 的场景，便于复用。
"""

import json
import re
from typing import Any


def parse_llm_output_to_dict(raw: str) -> dict[str, Any] | None:
    """
    将 LLM 回复文本解析为字典。
    会去掉 markdown 代码块（如 ```json ... ```），并尝试解析整段或首个 {} 对象。
    供调用方用 .get(key) 提取所需字段；其他 LLM 回答也可复用此函数。
    :param raw: LLM 返回的原始文本（可能含 JSON 或代码块包裹）
    :return: 解析得到的字典，失败返回 None
    """
    if not (raw and raw.strip()):
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except (json.JSONDecodeError, TypeError, ValueError):
                        break
    return None
