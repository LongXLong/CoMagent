# -*- coding: utf-8 -*-
"""
Insight Agent：收集多 Agent 阶段输出，调用 LLM 综合问题、Student 回答与各 Agent 反馈，
生成下一轮应关注与修改的总结，指导 Student 改进回答。
"""

from typing import Any

from config import OPENAI_MODEL
from utils.llm import llm_call
from utils.logger import get_logger
from utils.parse_llm_json import parse_llm_output_to_dict

logger = get_logger(__name__)


class InsightAgent:
    """
    综合多 Agent 反馈与当前回答，通过 LLM 生成下一轮改进指导。
    输出：下一轮 Student 应关注的问题、注意事项与修改建议的总结。
    """

    def integrate_feedback(
        self,
        question: str,
        base_answer: str,
        feedbacks: list[dict[str, Any]],
        *,
        polish_only: bool = False,
        expert_anchor: str | None = None,
    ) -> str:
        """
        收集多 Agent 阶段的输出，结合问题与 Student 当前回答，
        调用 LLM 做综合总结，明确下一轮应关注什么、注意什么、需要修改什么。
        返回给 Student 作为下一轮 revise 的指导文本。

        :param polish_only: 为 True 时用于 Expert 对齐之后：当前答案在事实上已定稿，
            仅从 General 反馈中抽取**表述/结构**类润色点，忽略任何与事实、数值、结论相冲突的建议。
        :param expert_anchor: polish_only 时传入阶段一 Expert 指导摘要，供模型识别「已对齐内容」。
        """
        if polish_only:
            return self._integrate_polish_only(
                question,
                base_answer,
                feedbacks,
                expert_anchor=(expert_anchor or "").strip() or None,
            )

        # 1. 收集多 Agent 阶段的输出
        expert_lines = []
        general_lines = []
        for f in feedbacks:
            agent = f.get("agent", "unknown")
            comment = (f.get("comment") or "").strip()
            if comment:
                line = f"【{agent}】\n{comment}"
                if str(agent).endswith("_expert"):
                    expert_lines.append(line)
                else:
                    general_lines.append(line)
        ordered_lines = expert_lines + general_lines
        raw_feedback = "\n\n".join(ordered_lines) if ordered_lines else "（各 Agent 暂无具体文字反馈。）"
        expert_feedback = "\n\n".join(expert_lines) if expert_lines else "（本轮无专家 Agent 有效反馈）"
        general_feedback = "\n\n".join(general_lines) if general_lines else "（本轮无通用 Agent 有效反馈）"
        logger.info(
            "Insight 收集到 %s 条 Agent 反馈，其中 expert=%s general=%s",
            len(ordered_lines),
            len(expert_lines),
            len(general_lines),
        )

        # 2. 若无有效反馈，返回默认提示，不调用 LLM
        if not ordered_lines:
            return "各 Agent 暂无具体反馈意见，请在表述严谨性、完整性与逻辑清晰度上稍作检查与优化。"

        # 3. 调用 LLM 综合问题、当前回答与多 Agent 反馈，生成下一轮改进总结（JSON）
        prompt = '''You are an expert "Reflection & Improvement Advisor" specialized in analyzing LLM-generated answers. Your sole purpose is to help improve the quality, accuracy, reasoning depth, completeness, safety, and user-value of the next iteration of the answer.

You will be given exactly three pieces of information:

【Question】
{question}

【Current Student Answer】
{base_answer}

【Expert-Agent Feedback】
{expert_feedback}

【General-Agent Feedback】
{general_feedback}

【All Feedback In Original Order】
{raw_feedback}

Your task is to deeply analyze the above three parts together, then distill the **most impactful and highest-priority improvement directions** for the next version of the answer.

Follow these strict analysis guidelines:

• Treat expert-agent feedback as higher-priority evidence than general-agent feedback, especially on factual correctness, calculations, domain constraints, and direct contradiction checks.
• If expert feedback conflicts with general-agent feedback, prefer the expert feedback unless the expert feedback is clearly unsupported by the question.
• Use general-agent feedback mainly for clarity, structure, completeness, tone, and explanatory quality when it does not conflict with expert guidance.
• Prioritize issues that affect factual correctness, logical coherence, major omissions, or severe misalignments with the question most highly.
• Give strong weight to feedback that appears repeatedly or comes from multiple agents.
• Identify any safety, bias, toxicity, overconfidence, hallucination, or misleading statement risks.
• Consider clarity, conciseness, structure, professional tone, and usefulness to the end user.
• Distinguish between "must-fix" problems and "nice-to-have" polish.
• Think about what specific evidence or reasoning is missing that could strengthen the answer.
• If the question is a closed-form problem and the question text already provides enough information, stay grounded in the question text only.
• In such closed-form cases, ignore reviewer suggestions that ask for external evidence, real-world statistics, extra assumptions, edge cases, individual differences, industry background, or scenario analysis not required by the question.
• Do not convert a self-contained question into an open-ended discussion. Prefer corrections and explanations that are directly derivable from the question.

Output **exclusively** a valid JSON object with **exactly** the following three keys. Do not include any other text, comments, markdown, explanations, apologies or code fences before/after the JSON.

{{
  "focus_points": [
    {{"source": "expert|general|mixed", "text": "short, clear, high-priority improvement area"}},
    {{"source": "expert|general|mixed", "text": "example: Correct factual error about X mentioned in feedback"}}
  ],
  "cautions": [
    {{"source": "expert|general|mixed", "text": "critical thing the next answer MUST avoid"}},
    {{"source": "expert|general|mixed", "text": "example: Do not repeat the hallucinated statistic about Y"}}
  ],
  "revision_suggestions": [
    {{"source": "expert|general|mixed", "text": "concrete, actionable improvement method or content direction"}},
    {{"source": "expert|general|mixed", "text": "example: Add a step-by-step reasoning chain to demonstrate how conclusion is reached"}}
  ]
}}

• Each array should contain 2–6 items (aim for precision, not quantity).
• Use concise, professional English.
• Items in each list should be independent, specific and non-redundant.
• Order the items in each array from most important → least important.
• When expert feedback exists, the first 1-2 focus points and the first caution should reflect the expert feedback unless it is clearly unsupported by the question.
• If expert feedback says the current answer is directionally correct and mainly needs clarification, do not exaggerate general-agent concerns into a full rewrite.
• Use general-agent feedback mainly to improve clarity, organization, completeness, and wording after expert concerns have been addressed.
• Every item must include a `source` field with one of: `expert`, `general`, or `mixed`.
• Use `expert` when the point primarily comes from expert-agent feedback.
• Use `general` when the point primarily comes from general-agent feedback.
• Use `mixed` only when the point genuinely combines both and cannot be attributed mainly to one side.

Output only the JSON.
'''.format(
            question=question.strip(),
            base_answer=(base_answer or "").strip(),
            expert_feedback=expert_feedback,
            general_feedback=general_feedback,
            raw_feedback=raw_feedback,
        )
        try:
            raw = llm_call(prompt, model=OPENAI_MODEL, temperature=0.0).strip()
            obj = parse_llm_output_to_dict(raw)
            if obj is not None:
                if not expert_lines:
                    obj = _normalize_guidance_sources(obj, force_source="general")
                guidance = _format_improvement_guidance(obj)
                if guidance:
                    if expert_lines:
                        guidance = (
                            "【反馈使用优先级】\n"
                            "1. 若专家反馈与通用反馈冲突，优先参考专家反馈。\n"
                            "2. 通用反馈主要用于补充清晰度、结构、完整性与表达，不应覆盖专家对事实、逻辑和任务要求的判断。\n\n"
                            + guidance
                        )
                    logger.info("Insight LLM 综合完成，指导长度=%s 字", len(guidance))
                    return guidance
            return raw if raw else raw_feedback
        except Exception:
            return raw_feedback

    def _integrate_polish_only(
        self,
        question: str,
        base_answer: str,
        feedbacks: list[dict[str, Any]],
        *,
        expert_anchor: str | None,
    ) -> str:
        """Expert 对齐后专用：General 反馈只产生润色指导，不得推动改结论或改数。"""
        general_lines: list[str] = []
        for f in feedbacks:
            agent = f.get("agent", "unknown")
            comment = (f.get("comment") or "").strip()
            if comment:
                general_lines.append(f"【{agent}】\n{comment}")
        if not general_lines:
            return ""

        raw_general = "\n\n".join(general_lines)
        anchor_block = (expert_anchor or "").strip() or "（Expert 已完成事实与推理对齐；当前学生答案中的最终结论与数值为真值来源。）"

        prompt = """You are a strict copy-editing advisor. The student's answer has ALREADY been factually aligned by an Expert stage.

HARD RULES (violate any → your output must be empty polish_points):
• Treat 【Current Student Answer】 as the single source of truth for final numbers, units, conclusions, and problem interpretation.
• General-agent feedback often suggests clarity fixes but MAY wrongly imply recalculations or different final answers. NEVER carry those into your output.
• Output ONLY safe polish: clearer sectioning, bullet/numbering, transitions, grammar, conciseness, explicit step labels—without changing any numeric result, equation outcome, or final verdict.
• If every general comment is purely factual/calculation critique that would change the answer, return an empty polish_points list.

【Question】
{question}

【Expert alignment context (already applied; do not contradict)】
{expert_anchor}

【Current Student Answer (preserve all factual content verbatim in meaning)】
{base_answer}

【General-Agent Feedback (filter; ignore anything that would alter facts or final numbers)】
{raw_general}

Return JSON only with exactly this shape:
{{
  "polish_points": [
    {{"text": "one stylistic or organizational suggestion only"}}
  ]
}}

Use concise English for each text field. If there is nothing safe to polish, return {{"polish_points": []}}.
Output only the JSON.""".format(
            question=question.strip(),
            expert_anchor=anchor_block,
            base_answer=(base_answer or "").strip(),
            raw_general=raw_general,
        )

        try:
            raw = llm_call(prompt, model=OPENAI_MODEL, temperature=0.0).strip()
            obj = parse_llm_output_to_dict(raw)
            if not isinstance(obj, dict):
                return ""
            points = obj.get("polish_points")
            if not isinstance(points, list) or not points:
                logger.info("Insight polish-only: 无安全润色点，跳过阶段二改写")
                return ""
            lines = [
                "【阶段二仅润色】以下仅涉及表述、分段、衔接与排版；禁止修改最终数值、单位、结论及已与 Expert 一致的推理路径。",
                "【可执行的润色点】",
            ]
            for i, item in enumerate(points, 1):
                if isinstance(item, dict):
                    t = str(item.get("text") or "").strip()
                else:
                    t = str(item or "").strip()
                if t:
                    lines.append(f"{i}. {t}")
            if len(lines) <= 2:
                return ""
            guidance = "\n".join(lines)
            logger.info("Insight polish-only 完成，指导长度=%s 字", len(guidance))
            return guidance
        except Exception as e:
            logger.warning("Insight polish-only 失败: %s", e)
            return ""

    def select_major_contributors(
        self,
        question: str,
        base_answer: str,
        feedbacks: list[dict[str, Any]],
        improved_answer: str,
        *,
        candidate_agents: list[str] | None = None,
        accepted: bool,
        improvement_score: float,
    ) -> list[str]:
        """
        Let an LLM judge which agents contributed most in the current round.
        Only evaluates the provided candidate general agents from the current round.
        """
        candidate_set = {str(name).strip() for name in (candidate_agents or []) if str(name).strip()}
        valid_agents = [
            str(f.get("agent", "")).strip()
            for f in feedbacks
            if str(f.get("agent", "")).strip() and str(f.get("agent", "")).strip() in candidate_set
        ]
        if not valid_agents:
            return []

        feedback_lines = []
        for f in feedbacks:
            agent = str(f.get("agent", "")).strip()
            comment = str(f.get("comment", "")).strip()
            if agent in candidate_set:
                feedback_lines.append(f"- {agent}: {comment or 'N/A'}")
        raw_feedback = "\n".join(feedback_lines)

        prompt = """You are evaluating which agents made the most meaningful contribution in one reflection round.

You will receive:
1. the user question
2. the student's answer before feedback
3. all agent feedback from this round
4. the student's answer after revision
5. whether the revised answer was accepted
6. the round improvement score

Select only the agents whose feedback made a clearly important contribution to the final round outcome.

Rules:
- Evaluate only the provided general-agent candidates. Ignore any expert agent feedback.
- Choose from the provided candidate agent names only.
- Prefer precision over recall: do not include an agent unless its feedback was clearly useful.
- If the revised answer was not accepted, return an empty list unless an agent still made a clearly strong diagnostic contribution.
- Select 1 or 2 agents only.
- If only one agent clearly stood out, return exactly 1.
- If two agents clearly made major contributions, return exactly 2.
- If no agent clearly stood out, return an empty list.

Question:
{question}

Student answer before feedback:
{base_answer}

Candidate general agents:
{candidate_agents}

Agent feedback:
{raw_feedback}

Student answer after revision:
{improved_answer}

Accepted:
{accepted}

Improvement score:
{improvement_score}

Return JSON only:
{{
  "contributor_agents": ["agent_name_1", "agent_name_2"],
  "reason": "short explanation"
}}""".strip().format(
            question=question.strip(),
            base_answer=(base_answer or "").strip(),
            candidate_agents=", ".join(valid_agents),
            raw_feedback=raw_feedback or "N/A",
            improved_answer=(improved_answer or "").strip(),
            accepted=str(bool(accepted)).lower(),
            improvement_score=f"{float(improvement_score):.4f}",
        )

        try:
            raw = llm_call(prompt, model=OPENAI_MODEL).strip()
            obj = parse_llm_output_to_dict(raw)
            if not isinstance(obj, dict):
                return []
            names = obj.get("contributor_agents")
            if not isinstance(names, list):
                return []
            valid_set = set(valid_agents)
            seen: set[str] = set()
            contributors: list[str] = []
            for item in names:
                name = str(item or "").strip()
                if name and name in valid_set and name not in seen:
                    seen.add(name)
                    contributors.append(name)
                if len(contributors) >= 2:
                    break
            logger.info(
                "Insight 贡献判定 accepted=%s improvement=%.3f contributors=%s reason=%s",
                accepted,
                improvement_score,
                contributors,
                str(obj.get("reason", "")).strip(),
            )
            return contributors
        except Exception as e:
            logger.warning("Insight 贡献判定失败: %s", e)
            return []


def _format_improvement_guidance(obj: dict[str, Any]) -> str:
    """将 LLM 返回的 JSON 格式化为供 Student 使用的改进指导文本。"""
    lines = []
    focus = obj.get("focus_points")
    if isinstance(focus, list) and focus:
        lines.append("【下一轮应优先关注】")
        for i, item in enumerate(focus, 1):
            source, text = _parse_guidance_item(item)
            prefix = f"[{source}] " if source else ""
            lines.append(f"{i}. {prefix}{text}")
    cautions = obj.get("cautions")
    if isinstance(cautions, list) and cautions:
        lines.append("\n【需避免或特别注意】")
        for i, item in enumerate(cautions, 1):
            source, text = _parse_guidance_item(item)
            prefix = f"[{source}] " if source else ""
            lines.append(f"{i}. {prefix}{text}")
    suggestions = obj.get("revision_suggestions")
    if isinstance(suggestions, list) and suggestions:
        lines.append("\n【具体修改建议】")
        for i, item in enumerate(suggestions, 1):
            source, text = _parse_guidance_item(item)
            prefix = f"[{source}] " if source else ""
            lines.append(f"{i}. {prefix}{text}")
    if not lines:
        return ""
    return "\n".join(lines).strip()


def _normalize_guidance_sources(obj: dict[str, Any], *, force_source: str | None = None) -> dict[str, Any]:
    if force_source not in {"expert", "general", "mixed", None}:
        force_source = None
    if force_source is None:
        return obj
    normalized: dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(value, list):
            new_items: list[Any] = []
            for item in value:
                if isinstance(item, dict):
                    new_item = dict(item)
                    new_item["source"] = force_source
                    new_items.append(new_item)
                else:
                    new_items.append({"source": force_source, "text": str(item or "").strip()})
            normalized[key] = new_items
        else:
            normalized[key] = value
    return normalized


def _parse_guidance_item(item: Any) -> tuple[str, str]:
    """解析 guidance item，兼容旧字符串格式和新对象格式。"""
    if isinstance(item, dict):
        source = str(item.get("source") or "").strip().lower()
        text = str(item.get("text") or "").strip()
        if source not in {"expert", "general", "mixed"}:
            source = ""
        return source, text
    return "", str(item or "").strip()
