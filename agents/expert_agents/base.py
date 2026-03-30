# -*- coding: utf-8 -*-
"""专家 Agent 基类：候选类别内优先精确检索，再做向量召回。"""

import re
from abc import ABC
from typing import Any

from agents.general_agents.base import AGENT_OUTPUT_JSON_SCHEMA, parse_agent_output
from memory.ltm import (
    infer_question_types_for_ltm,
    retrieve_knowledge_bundle_from_vector_store,
)
from utils.llm import llm_call
from utils.logger import get_logger
from utils.parse_llm_json import parse_llm_output_to_dict

logger = get_logger(__name__)

EXPERT_REVIEW_JSON_SCHEMA = """
请务必只输出一个合法的 JSON 对象，不要输出任何其他文字、解释或 Markdown 标记。
JSON 格式固定为：
{
    "verdict": "aligned|conflict|uncertain",
    "response_mode": "issues|detailed_solution",
    "issue_summary": "一句话说明当前回答与精确答案的一致或冲突情况",
    "logic_issues": [
        {
            "quoted_text": "从 student 当前回答中原样摘录出有逻辑错误的那句话或那一小段；若没有则为空字符串",
            "why_wrong": "这句话/这段逻辑为什么错",
            "correct_thought_points": [
                "正确思路第1点，只写这一处应该怎么想/怎么算",
                "正确思路第2点"
            ]
        }
    ],
    "detailed_solution_points": [
        "当 student 当前回答只有答案或只有一个算式，按正确解题顺序给出的详细思路第1点",
        "详细思路第2点"
    ],
    "style_note": "若结论已正确，可给出简短表述优化建议；否则可为空字符串"
}
不要输出 ```json 等代码块标记，只输出纯 JSON。
""".strip()


def _count_reasoning_markers(text: str) -> int:
    clean = str(text or "").lower()
    markers = [
        "=",
        "therefore",
        "thus",
        "because",
        "first",
        "then",
        "next",
        "finally",
        "calculate",
        "so ",
        "因此",
        "所以",
        "先",
        "然后",
        "接着",
        "最后",
        "计算",
        "步骤",
    ]
    return sum(1 for marker in markers if marker in clean)


def _answer_has_clear_steps(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return False
    sentence_like_parts = [
        part.strip()
        for part in re.split(r"[\n。！？!?;；]+", clean)
        if part.strip()
    ]
    if len(sentence_like_parts) >= 3:
        return True
    if _count_reasoning_markers(clean) >= 3 and len(clean) >= 60:
        return True
    return False


def _answer_is_short_or_single_expression(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return True
    if len(clean) <= 40 and len(clean.split()) <= 8:
        return True
    if "\n" not in clean:
        math_chars = re.sub(r"[0-9\.\+\-\*/=()%,$<> ]+", "", clean)
        if len(math_chars) <= 8 and "=" in clean:
            return True
    return False


def _canonical_answer_to_points(canonical_answer: str) -> list[str]:
    raw = str(canonical_answer or "").strip()
    if not raw:
        return []
    pieces: list[str] = []
    for chunk in re.split(r"\n+", raw):
        text = chunk.strip()
        if not text or text.startswith("####"):
            continue
        pieces.append(text)
    if not pieces:
        for chunk in re.split(r"(?<=[。！？!?\.])\s+", raw):
            text = chunk.strip()
            if text and not text.startswith("####"):
                pieces.append(text)
    normalized: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        if piece not in seen:
            seen.add(piece)
            normalized.append(piece)
        if len(normalized) >= 6:
            break
    return normalized


def _parse_expert_review_json(raw: str) -> dict[str, Any] | None:
    obj = parse_llm_output_to_dict(raw or "")
    if not isinstance(obj, dict):
        return None
    verdict = str(obj.get("verdict") or "").strip().lower()
    if verdict not in {"aligned", "conflict", "uncertain"}:
        verdict = "uncertain"
    response_mode = str(obj.get("response_mode") or "").strip().lower()
    if response_mode not in {"issues", "detailed_solution"}:
        response_mode = "issues"
    parsed_issues: list[dict[str, Any]] = []
    logic_issues = obj.get("logic_issues")
    if isinstance(logic_issues, list):
        for item in logic_issues[:4]:
            if not isinstance(item, dict):
                continue
            quoted_text = str(item.get("quoted_text") or "").strip()
            why_wrong = str(item.get("why_wrong") or "").strip()
            correct_points_raw = item.get("correct_thought_points")
            correct_points: list[str] = []
            if isinstance(correct_points_raw, list):
                for point in correct_points_raw[:4]:
                    text = str(point or "").strip()
                    if text:
                        correct_points.append(text)
            if quoted_text or why_wrong or correct_points:
                parsed_issues.append(
                    {
                        "quoted_text": quoted_text,
                        "why_wrong": why_wrong,
                        "correct_thought_points": correct_points,
                    }
                )
    detailed_solution_points: list[str] = []
    solution_points_raw = obj.get("detailed_solution_points")
    if isinstance(solution_points_raw, list):
        for point in solution_points_raw[:6]:
            text = str(point or "").strip()
            if text:
                detailed_solution_points.append(text)
    return {
        "verdict": verdict,
        "response_mode": response_mode,
        "issue_summary": str(obj.get("issue_summary") or "").strip(),
        "logic_issues": parsed_issues,
        "detailed_solution_points": detailed_solution_points,
        "style_note": str(obj.get("style_note") or "").strip(),
    }


def _build_expert_comment(
    parsed: dict[str, Any] | None,
    *,
    exact_found: bool,
    canonical_answer: str,
    fallback_comment: str,
) -> str:
    if not exact_found:
        return fallback_comment
    if not parsed:
        return fallback_comment
    verdict = parsed.get("verdict", "uncertain")
    response_mode = parsed.get("response_mode", "issues")
    issue_summary = parsed.get("issue_summary", "").strip()
    logic_issues = parsed.get("logic_issues") or []
    detailed_solution_points = parsed.get("detailed_solution_points") or []
    style_note = parsed.get("style_note", "").strip()
    parts: list[str] = []
    if verdict == "aligned":
        parts.append("当前回答与精确检索命中的标准答案一致。")
        if issue_summary:
            parts.append(issue_summary)
        if style_note:
            parts.append(style_note)
        return "\n".join(x for x in parts if x).strip() or fallback_comment
    elif verdict == "conflict":
        parts.append("当前回答与精确检索命中的标准答案冲突。请优先修正下列逻辑错误。")
    else:
        parts.append("请优先对照精确检索命中的标准答案逻辑核对当前回答，并定位出错语句。")
    if issue_summary:
        parts.append(issue_summary)
    if response_mode == "detailed_solution" and isinstance(detailed_solution_points, list) and detailed_solution_points:
        parts.append("当前回答信息过少或只有单个算式，按下列详细思路重做：")
        for idx, point in enumerate(detailed_solution_points, 1):
            text = str(point or "").strip()
            if text:
                parts.append(f"{idx}. {text}")
    elif isinstance(logic_issues, list) and logic_issues:
        for idx, issue in enumerate(logic_issues, 1):
            if not isinstance(issue, dict):
                continue
            quoted_text = str(issue.get("quoted_text") or "").strip()
            why_wrong = str(issue.get("why_wrong") or "").strip()
            correct_points = issue.get("correct_thought_points") or []
            issue_lines = [f"问题{idx}："]
            issue_lines.append(f"原句/原段：{quoted_text}" if quoted_text else "原句/原段：需要定位该处错误语句")
            issue_lines.append(f"原因：{why_wrong}" if why_wrong else "原因：该处逻辑与精确检索命中的标准答案不一致。")
            if isinstance(correct_points, list) and correct_points:
                point_text = " ".join(
                    f"{point_idx}. {str(point).strip()}"
                    for point_idx, point in enumerate(correct_points, 1)
                    if str(point).strip()
                ).strip()
                if point_text:
                    issue_lines.append(f"正确思路：{point_text}")
            else:
                issue_lines.append("正确思路：请按精确检索命中的标准答案逻辑重做这一处。")
            parts.append("\n".join(issue_lines).strip())
    if style_note:
        parts.append(style_note)
    return "\n".join(x for x in parts if x).strip() or fallback_comment

class BaseExpertAgent(ABC):
    agent_name: str = "base_expert"
    question_type: str = "OTHER_GENERAL_Q"
    expert_title: str = "领域专家"
    review_items: tuple[str, ...] = (
        "回答是否准确覆盖该领域核心概念？",
        "回答是否存在领域常见错误、逻辑跳跃或不严谨表述？",
        "请给出可执行的改进建议，提升专业性与可靠性。",
    )
    default_comment: str = "可从该领域专业标准补充与修正回答。"
    default_score: float = 0.78

    def retrieve_knowledge_bundle(
        self,
        question: str,
        candidate_question_types: list[str] | None = None,
    ) -> dict[str, Any]:
        search_types: list[str] = []
        for item in candidate_question_types or []:
            qt = (item or "").strip().upper()
            if qt and qt not in search_types:
                search_types.append(qt)
        if not candidate_question_types:
            search_types = infer_question_types_for_ltm(question, top_n=2)
        if not search_types:
            search_types = [self.question_type]
        search_types = search_types[:2]
        bundle = retrieve_knowledge_bundle_from_vector_store(
            question,
            search_types,
        )
        hit_count = int(bundle.get("hit_count") or 0)
        knowledge = str(bundle.get("knowledge") or "")
        exact_found = bool(bundle.get("exact_found"))
        if hit_count > 0:
            knowledge_rows = []
            if knowledge:
                for block in knowledge.split("\n\n")[:3]:
                    preview = block.replace("\n", " ")[:100]
                    if preview:
                        knowledge_rows.append(preview)
            logger.info(
                "[TRACE] expert=%s question_type=%s search_types=%s retrieval=%s hits=%s hit_questions=%s",
                self.agent_name,
                self.question_type,
                search_types,
                "exact" if exact_found else "vector",
                hit_count,
                knowledge_rows,
            )
            return bundle
        logger.info(
            "[TRACE] expert=%s question_type=%s search_types=%s retrieval=miss hits=0",
            self.agent_name,
            self.question_type,
            search_types,
        )
        return bundle

    def review(
        self,
        answer: str,
        question: str = "",
        candidate_question_types: list[str] | None = None,
        cached_knowledge_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        bundle = cached_knowledge_bundle or self.retrieve_knowledge_bundle(
            question,
            candidate_question_types,
        )
        domain_knowledge = str(bundle.get("knowledge") or "")
        exact_found = bool(bundle.get("exact_found"))
        canonical_answer = str(bundle.get("canonical_answer") or "").strip()
        checks = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(self.review_items))
        answer_has_clear_steps = _answer_has_clear_steps(answer)
        answer_is_short = _answer_is_short_or_single_expression(answer)
        preferred_response_mode = "detailed_solution" if answer_is_short and not answer_has_clear_steps else "issues"
        authority_instruction = (
            "以下 knowledge 是对当前问题的精确检索命中，其中的 answer 必须被视为权威依据。"
            "如果待审查回答与该 answer 有任何冲突、偏离、遗漏或额外臆造内容，你必须明确指出，"
            "并要求严格按精确检索出的 answer 修正。不得弱化、忽略或改写这条精确答案的约束力。"
            "你不得重新推导、改写或发明新的最终答案；若需要引用正确结论，只能直接引用 knowledge 中已有的 answer。"
            if exact_found
            else "当前未检索到该问题的精确 knowledge。你可以基于领域角色做审查，但不得伪造或臆测知识库中存在的标准答案。"
        )
        prompt = """
你是一位“{expert_title}”。请对回答进行严格的领域审查。

{authority_instruction}

你只能参考以下“本领域 knowledge”，不得引用其他领域知识：
{knowledge}

原始问题：
{question}

待审查回答：
{answer}

请检查：
{checks}

如果存在精确检索命中：
1. 只能判断当前回答是否与精确答案一致、冲突或无法确定。
2. 不要自己重新计算，不要引入 knowledge 里没有的新数值、新结论或新推导。
3. 不要把完整标准答案、完整标准解题过程直接重复给 student。
4. 你的核心任务是像批改作业一样，指出当前回答中具体哪几句话或哪几小段逻辑错了；`quoted_text` 必须尽量直接摘录 student 当前回答里的错误原句或原段。
5. 对每个错误点，用 `correct_thought_points` 分点列出正确思路，告诉 student 这一处应该怎么想、怎么算、或先后关系应该如何处理。
6. `correct_thought_points` 只能写与该错误点直接相关的纠正思路，不要泄露整道题的完整标准解答。
7. 若当前回答结论已与精确答案一致，只能给出轻量的表述/结构优化建议。
8. 如果当前回答只有最终答案、只有一个算式、或明显缺少可批改的中间步骤，请把 `response_mode` 设为 `detailed_solution`，并在 `detailed_solution_points` 中给出按正确顺序展开的详细解题思路。
9. 如果当前回答本身已经包含了较清晰的分步推理，请把 `response_mode` 设为 `issues`，并按“问题1/问题2”方式指出具体出错语句与对应修正思路。

当前回答形态提示：
- has_clear_steps = {answer_has_clear_steps}
- is_short_or_single_expression = {answer_is_short}
- preferred_response_mode = {preferred_response_mode}

{json_schema}
""".strip().format(
            expert_title=self.expert_title,
            authority_instruction=authority_instruction,
            knowledge=domain_knowledge or "（当前候选类别中未检索到该问题的精确 knowledge）",
            question=question,
            answer=answer,
            checks=checks,
            answer_has_clear_steps=str(answer_has_clear_steps).lower(),
            answer_is_short=str(answer_is_short).lower(),
            preferred_response_mode=preferred_response_mode,
            json_schema=(EXPERT_REVIEW_JSON_SCHEMA if exact_found else AGENT_OUTPUT_JSON_SCHEMA).strip(),
        )
        raw = llm_call(prompt)
        fallback_comment = (
            "已检索到当前问题的精确答案。请根据当前回答的形态，若有清晰步骤则逐条指出错误语句；若只有答案或单个算式，则给出详细正确思路。"
            if exact_found
            else self.default_comment
        )
        if exact_found:
            parsed = _parse_expert_review_json(raw)
            if parsed is not None:
                if parsed.get("verdict") == "aligned":
                    parsed["logic_issues"] = []
                    parsed["detailed_solution_points"] = []
                elif preferred_response_mode == "detailed_solution":
                    parsed["response_mode"] = "detailed_solution"
                    parsed["logic_issues"] = []
                    if not parsed.get("detailed_solution_points"):
                        parsed["detailed_solution_points"] = _canonical_answer_to_points(canonical_answer)
            comment_text = _build_expert_comment(
                parsed,
                exact_found=exact_found,
                canonical_answer=canonical_answer,
                fallback_comment=fallback_comment,
            )
            score = self.default_score
        else:
            comment_text, score = parse_agent_output(raw, self.default_score)
        return {
            "agent": self.agent_name,
            "question_type": self.question_type,
            "comment": comment_text or fallback_comment,
            "score": score,
            "exact_found": exact_found,
            "expert_verdict": parsed.get("verdict", "") if exact_found and parsed else "",
            "expert_response_mode": parsed.get("response_mode", "") if exact_found and parsed else "",
        }
