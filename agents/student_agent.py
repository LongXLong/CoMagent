# -*- coding: utf-8 -*-
"""Student Agent：候选类别内向量检索 + 生成回答 + 迭代修订。"""

import re
from typing import Any

from memory.ltm import (
    format_qa_context,
    infer_question_type_for_ltm,
    infer_question_types_for_ltm,
    vector_search_qa_scored,
)
from utils.llm import llm_call, semantic_similarity
from utils.logger import get_logger
from utils.parse_llm_json import parse_llm_output_to_dict

logger = get_logger(__name__)

STUDENT_TOP_CATEGORY_COUNT = 2
STUDENT_VECTOR_TOP_K = 6
STUDENT_MIN_SIMILARITY = 0.55
STUDENT_STRICT_FALLBACK_SIMILARITY = 0.82
STUDENT_USE_LTM_RETRIEVAL = False


def _parse_answer_json(raw: str) -> str:
    obj = parse_llm_output_to_dict(raw or "")
    if obj is not None:
        ans = obj.get("answer")
        return str(ans or "").strip()
    return str(raw or "").strip()


def _parse_accept_json(raw: str) -> tuple[bool | None, str]:
    obj = parse_llm_output_to_dict(raw or "")
    if obj is None:
        return None, ""
    accept = obj.get("accept_candidate")
    reason = str(obj.get("reason") or "").strip()
    if isinstance(accept, bool):
        return accept, reason
    return None, reason


def _is_low_information_answer(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return True
    words = re.findall(r"\S+", clean)
    return len(clean) <= 40 or len(words) <= 8


def _has_reasoning_markers(text: str) -> bool:
    clean = str(text or "").lower()
    markers = [
        "=",
        "therefore",
        "thus",
        "so ",
        "because",
        "first",
        "then",
        "finally",
        "calculate",
        "calculation",
        "steps",
        "因此",
        "所以",
        "先",
        "然后",
        "最后",
        "计算",
        "步骤",
    ]
    return any(marker in clean for marker in markers)


def _has_uncertainty_or_speculation(text: str) -> bool:
    clean = str(text or "").lower()
    markers = [
        "may ",
        "might ",
        "maybe",
        "possibly",
        "perhaps",
        "at least",
        "at most",
        "cannot determine",
        "can't determine",
        "without further information",
        "without more information",
        "depends",
        "assume",
        "assuming",
        "可能",
        "也许",
        "或许",
        "无法确定",
        "取决于",
        "假设",
    ]
    return any(marker in clean for marker in markers)


def _reviewer_guidance_signals_issue(text: str) -> bool:
    clean = str(text or "").lower()
    markers = [
        "incorrect",
        "incomplete",
        "wrong",
        "missing",
        "does not",
        "lack",
        "recalculate",
        "logic jump",
        "逻辑跳跃",
        "缺证据",
        "不正确",
        "错误",
        "缺少",
        "重新计算",
    ]
    return any(marker in clean for marker in markers)


def _extract_last_number(text: str) -> str:
    nums = re.findall(r"-?\d+(?:\.\d+)?%?", str(text or ""))
    return nums[-1] if nums else ""


def _fallback_accept_candidate(
    current_answer: str,
    candidate_answer: str,
    reviewer_guidance: str,
) -> tuple[bool | None, str]:
    current_low_info = _is_low_information_answer(current_answer)
    candidate_reasoned = _has_reasoning_markers(candidate_answer)
    candidate_risky = _has_uncertainty_or_speculation(candidate_answer)
    guidance_signals_issue = _reviewer_guidance_signals_issue(reviewer_guidance)
    same_final_number = False
    current_last_num = _extract_last_number(current_answer)
    candidate_last_num = _extract_last_number(candidate_answer)
    if current_last_num and candidate_last_num and current_last_num == candidate_last_num:
        same_final_number = True

    if candidate_risky:
        return False, "fallback: 候选包含明显不确定或推测表达，保留当前答案。"

    if same_final_number and current_low_info:
        return False, "fallback: 候选主要是在展开相同结论，保留更简洁的当前答案。"

    if current_low_info and candidate_reasoned and guidance_signals_issue:
        return True, "fallback: 主验收解析失败，但当前答案信息过少且反馈显示存在问题，接受更完整的候选答案。"

    if current_low_info and candidate_reasoned and len(candidate_answer) >= max(80, len(current_answer) * 3):
        return True, "fallback: 主验收解析失败，当前答案过短，候选提供了更完整且更直接的解答。"

    return None, ""


def _parse_applicability_results(raw: str) -> dict[int, tuple[bool, str]]:
    obj = parse_llm_output_to_dict(raw or "")
    if not isinstance(obj, dict):
        return {}
    results = obj.get("results")
    if not isinstance(results, list):
        return {}

    out: dict[int, tuple[bool, str]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        applicable = item.get("applicable")
        reason = str(item.get("reason") or "").strip()
        if isinstance(idx, int) and isinstance(applicable, bool):
            out[idx] = (applicable, reason)
    return out


def _filter_applicable_rows(question: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    candidate_blocks = []
    for idx, row in enumerate(rows):
        candidate_blocks.append(
            "\n".join(
                [
                    f"Index: {idx}",
                    f"Category: {row.get('category', '')}",
                    f"Similarity: {float(row.get('similarity', 0.0)):.4f}",
                    f"Stored Question: {row.get('question', '')}",
                    f"Stored Answer: {row.get('answer', '')}",
                ]
            )
        )

    prompt = """
You are a strict retrieval applicability judge.
Decide whether each retrieved QA memory can be safely used as direct support for answering the current question.

Keep a memory only if it is truly applicable to the same task, constraints, and answer target.
Reject memories that are only topically similar, partially related, or likely to bias the answer.

Current question:
{question}

Candidate memories:
{candidate_memories}

Return JSON only:
{{
  "results": [
    {{"index": 0, "applicable": true, "reason": "short reason"}}
  ]
}}
""".strip().format(
        question=question,
        candidate_memories="\n\n".join(candidate_blocks),
    )
    raw = llm_call(
        prompt,
        system_prompt="You are a strict retrieval filter. Return valid JSON only.",
    )
    applicability = _parse_applicability_results(raw)
    if not applicability:
        fallback_rows = [
            row
            for row in rows
            if float(row.get("similarity", 0.0)) >= STUDENT_STRICT_FALLBACK_SIMILARITY
        ]
        logger.warning(
            "Student 适用性判断解析失败，回退高阈值过滤 fallback_hits=%s",
            len(fallback_rows),
        )
        return fallback_rows

    kept_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        applicable, reason = applicability.get(idx, (False, "missing_result"))
        logger.info(
            "Student 适用性判断 idx=%s applicable=%s similarity=%.4f reason=%s",
            idx,
            applicable,
            float(row.get("similarity", 0.0)),
            reason,
        )
        if applicable:
            kept_rows.append(row)
    return kept_rows


class StudentAgent:
    """负责：候选类别内检索后生成初答，并在循环中根据反馈修订答案。"""

    def answer(
        self,
        question: str,
        ltm: dict[str, Any],
        mk: dict[str, Any],
    ) -> tuple[str, str, list[str]]:
        candidate_question_types = infer_question_types_for_ltm(
            question,
            top_n=STUDENT_TOP_CATEGORY_COUNT,
        )
        question_type = candidate_question_types[0] if candidate_question_types else infer_question_type_for_ltm(question)
        if STUDENT_USE_LTM_RETRIEVAL:
            retrieved_rows = vector_search_qa_scored(
                ltm,
                question,
                top_k=STUDENT_VECTOR_TOP_K,
                question_types=candidate_question_types,
            )
            threshold_rows = [
                row
                for row in retrieved_rows
                if float(row.get("similarity", 0.0)) >= STUDENT_MIN_SIMILARITY
            ]
            applicable_rows = _filter_applicable_rows(question, threshold_rows)
            retrieved_knowledge = format_qa_context(applicable_rows)
            raw_categories = [x.get("category", "") for x in retrieved_rows]
            kept_categories = [x.get("category", "") for x in applicable_rows]
            logger.info(
                (
                    "Student 检索完成 primary_question_type=%s candidate_question_types=%s "
                    "raw_hits=%s threshold_hits=%s applicable_hits=%s raw_categories=%s kept_categories=%s"
                ),
                question_type,
                candidate_question_types,
                len(retrieved_rows),
                len(threshold_rows),
                len(applicable_rows),
                raw_categories,
                kept_categories,
            )
            logger.info(
                "Student 检索相似度 raw=%s threshold=%.2f",
                [round(float(x.get("similarity", 0.0)), 4) for x in retrieved_rows],
                STUDENT_MIN_SIMILARITY,
            )
        else:
            retrieved_rows = []
            threshold_rows = []
            applicable_rows = []
            retrieved_knowledge = "（Student 侧已禁用 LTM 检索，仅基于自身能力作答）"
            logger.info(
                "Student LTM 检索已禁用 primary_question_type=%s candidate_question_types=%s",
                question_type,
                candidate_question_types,
            )
        prompt = """
The following are retrieved QA memories:
{retrieved_knowledge}

Use a retrieved memory only when it is directly applicable to the current question.
If retrieved memories are insufficient or not directly applicable, answer with your own capability, but remain rigorous.
Do not copy a retrieved answer if the task, constraints, or answer target do not fully match.

Only output JSON:
{{
  "answer": "your answer"
}}

Question: {question}
""".strip().format(retrieved_knowledge=retrieved_knowledge, question=question)
        raw = self.generate_response(prompt)
        answer = _parse_answer_json(raw)
        logger.info(
            "Student 初答完成 question_type=%s candidate_question_types=%s 答案长度=%s",
            question_type,
            candidate_question_types,
            len(answer or ""),
        )
        return answer, question_type, candidate_question_types

    def generate_response(self, text: str) -> str:
        return llm_call(text)

    def revise_answer(
        self,
        question: str,
        current_answer: str,
        organized_feedback: str,
        *,
        stage: str = "standard",
        canonical_answer: str = "",
    ) -> str:
        stage = (stage or "standard").strip().lower()
        canonical_answer = (canonical_answer or "").strip()
        if stage == "expert_alignment":
            stage_instruction = """
You are in Stage 1: expert alignment.

Use only the provided expert guidance to correct the answer.
Your job is to fix factual errors, calculations, units, constraints, and the final conclusion.
Do not perform stylistic polishing beyond what is necessary to make the answer correct and direct.
The expert guidance should be treated as a grading-style diagnosis of which steps are wrong and how those steps should be corrected.
Do not copy or invent a full standard solution that was not explicitly provided in the feedback.
Prefer direct correction over paraphrasing.
""".strip()
        elif stage == "general_polish":
            stage_instruction = """
You are in Stage 2: general polish after Expert alignment.

The current answer is already factually locked to the Expert stage. Your only job is light editing.

Rules (non-negotiable):
• Keep the same final numeric results, units, yes/no conclusions, and the same problem interpretation (e.g. what counts as “remaining work” after a restart) as in the current answer.
• Do not add, remove, or reorder calculation steps in a way that changes any intermediate or final values.
• If the feedback asks to recalculate, change the final answer, or adopt a different modeling of the scenario, ignore that part of the feedback entirely.
• Apply only safe changes: clearer headings, bullets/numbering, transitions, grammar, conciseness, or splitting paragraphs—without altering math or logic outcomes.

If the feedback is empty or only asks for disallowed factual changes, return the current answer unchanged (same meaning and numbers).

If a canonical answer reference is provided, stay consistent with it and with the current answer’s conclusions.
""".strip()
        else:
            stage_instruction = """
Your goal is to improve the answer conservatively.
Keep correct parts of the current answer unless the feedback clearly shows they should change.
Do not introduce new assumptions, external facts, edge cases, or speculative content unless they are required by the question.
If the feedback is weak, conflicting, or not clearly actionable, stay close to the current answer.
Prefer minimal edits that fix the highest-priority issues first.
The revised answer must still answer the original question directly.
""".strip()

        prompt = """
You revise an answer using reviewer feedback.

{stage_instruction}

Question:
{question}

Current answer:
{current_answer}

Canonical answer reference:
{canonical_answer}

Feedback:
{organized_feedback}

Return JSON only:
{{
  "answer": "revised answer"
}}
""".strip().format(
            stage_instruction=stage_instruction,
            question=question,
            current_answer=current_answer,
            canonical_answer=canonical_answer or "N/A",
            organized_feedback=organized_feedback,
        )
        raw = self.generate_response(prompt)
        obj = parse_llm_output_to_dict(raw)
        if obj is not None and obj.get("answer") is not None:
            out = str(obj.get("answer") or "").strip()
            logger.info("Student revise_answer 输出长度=%s", len(out))
            return out
        return _parse_answer_json(raw) if raw else ""

    def choose_better_answer(
        self,
        question: str,
        current_answer: str,
        candidate_answer: str,
        reviewer_guidance: str = "",
    ) -> tuple[str, dict[str, Any]]:
        current_clean = (current_answer or "").strip()
        candidate_clean = (candidate_answer or "").strip()
        if not candidate_clean:
            return current_clean, {
                "accepted": False,
                "reason": "候选答案为空，保留当前答案。",
            }
        if candidate_clean == current_clean:
            return current_clean, {
                "accepted": False,
                "reason": "候选答案与当前答案相同，无需替换。",
            }

        prompt = """
You are a conservative answer-quality judge.
Compare the current answer and a revised candidate for the same question.

Decision rule:
Accept the candidate only if it is clearly better overall.
If the candidate is only stylistically different, more verbose, speculative, or not clearly better, reject it.
Reject the candidate if it introduces unsupported assumptions, new risks, or drifts away from the question.

Focus on:
1. whether the answer directly addresses the question
2. factual and logical consistency
3. whether the candidate resolves important reviewer concerns
4. whether the candidate introduces new problems

Question:
{question}

Current answer:
{current_answer}

Candidate answer:
{candidate_answer}

Reviewer guidance:
{reviewer_guidance}

Return JSON only:
{{
  "accept_candidate": true,
  "reason": "short explanation"
}}
""".strip().format(
            question=question,
            current_answer=current_clean,
            candidate_answer=candidate_clean,
            reviewer_guidance=reviewer_guidance or "N/A",
        )
        raw = self.generate_response(prompt)
        accept, reason = _parse_accept_json(raw)
        if accept is None:
            accept, fallback_reason = _fallback_accept_candidate(
                current_clean,
                candidate_clean,
                reviewer_guidance,
            )
            if accept is None:
                accept = False
                reason = reason or "候选答案验收结果无法解析，默认保留当前答案。"
            else:
                reason = fallback_reason
        chosen = candidate_clean if accept else current_clean
        logger.info(
            "Student choose_better_answer accepted=%s reason=%s chosen_length=%s",
            accept,
            reason,
            len(chosen),
        )
        return chosen, {"accepted": accept, "reason": reason}

    def evaluate_answer(
        self,
        answer: str,
        ltm: dict[str, Any],
        *,
        initial_answer: str | None = None,
        previous_answer: str | None = None,
    ) -> tuple[float, float]:
        if initial_answer is None and previous_answer is None:
            consistency_score = 0.5
        else:
            ref = previous_answer if previous_answer else initial_answer
            consistency_score = semantic_similarity(answer, ref) if ref else 0.5
        confidence = 0.5 + 0.5 * consistency_score
        return confidence, consistency_score
