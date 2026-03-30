# -*- coding: utf-8 -*-
"""
多 Agent 反思系统主流程：初答 → MK 选 Agent（或用户指定）→ 多 Agent 反馈 → Insight 整合 → Student 修正 → 循环/输出 → 可选更新 LTM/MK。
"""

from typing import Any

from agents import ExpertAgentFactory, GeneralAgentFactory, InsightAgent, MetaKnowledge, StudentAgent
from utils.logger import get_logger

from memory import (
    create_wm,
    evolve_mk_from_better_agents,
    load_ltm,
    load_mk,
    save_mk,
    update_mk_from_ltm,
    update_wm,
)
from utils.llm import (
    compute_improvement,
    get_token_usage,
    reset_embedding_cache,
    reset_token_usage,
)

logger = get_logger(__name__)


def _feedbacks_to_text(feedbacks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for fb in feedbacks:
        agent = str(fb.get("agent") or "?").strip()
        comment = str(fb.get("comment") or "").strip()
        if comment:
            lines.append(f"【{agent}】\n{comment}")
    return "\n\n".join(lines).strip()


def _log_feedbacks(feedbacks: list[dict[str, Any]], *, prefix: str = "") -> None:
    for fb in feedbacks:
        comment = (fb.get("comment") or "").strip()
        if comment:
            logger.info("%s[%s] 反馈: %s", prefix, fb.get("agent", "?"), comment)


def get_suggest(question: str) -> dict[str, Any]:
    """
    根据问题返回 MK 建议的 Agent 列表与初答，供前端展示为系统建议勾选。
    返回: initial_answer, question_type, candidate_question_types, suggested_agents, all_agents
    """
    reset_embedding_cache()
    logger.info("get_suggest 开始 question=%s", question[:80] + "..." if len(question) > 80 else question)
    ltm = load_ltm()
    mk_data = load_mk()
    mk = MetaKnowledge(mk_data)
    student = StudentAgent()
    answer, question_type, candidate_question_types = student.answer(question, ltm, mk_data)
    logger.info(
        "[TRACE] suggest question_type=%s candidate_question_types=%s",
        question_type,
        candidate_question_types,
    )
    confidence, consistency = student.evaluate_answer(answer, ltm)
    suggested_agents = mk.select_agents(
        question, confidence, consistency, question_type=question_type
    )
    all_agents = GeneralAgentFactory.all_names()
    suggested_agents = [a for a in suggested_agents if a in all_agents]
    logger.info(
        "get_suggest 完成 question_type=%s candidate_question_types=%s suggested_agents=%s",
        question_type,
        candidate_question_types,
        suggested_agents,
    )
    return {
        "initial_answer": answer,
        "question_type": question_type,
        "candidate_question_types": candidate_question_types,
        "suggested_agents": suggested_agents,
        "all_agents": all_agents,
    }


def run_system(
    question: str,
    *,
    selected_agents: list[str] | None = None,
    do_update_ltm: bool = True,
    do_update_mk: bool = True,
    suggested_initial_answer: str | None = None,
    suggested_question_type: str | None = None,
    suggested_candidate_question_types: list[str] | None = None,
) -> dict[str, Any]:
    """
    主流程。
    :param question: 用户问题
    :param selected_agents: 兼容参数，当前不作为主选择依据
    :param do_update_ltm: 兼容参数，当前阶段不更新 LTM
    :param do_update_mk: 是否在结束后更新并保存 MK（含通用 Agent MAB 统计、update_mk_from_ltm）
    :param suggested_initial_answer: 若提供（非空），则复用 Get Suggest 阶段的初答，不再调用 ``student.answer``
    :param suggested_question_type: 与初答配套的问题类型（缺省为 ``general``）
    :param suggested_candidate_question_types: 候选问题类型列表（缺省为 ``[question_type]``）
    :return: {"initial_answer": 初答, "final_answer": 最终答案}
    """
    reset_embedding_cache()
    logger.info(
        "run_system 开始 question=%s selected_agents=%s do_update_ltm=%s do_update_mk=%s",
        question[:80] + "..." if len(question) > 80 else question,
        selected_agents,
        do_update_ltm,
        do_update_mk,
    )
    ltm = load_ltm()
    mk_data = load_mk()
    mk = MetaKnowledge(mk_data)
    student = StudentAgent()
    insight = InsightAgent()
    wm = create_wm()

    seed_text = (suggested_initial_answer or "").strip()
    if seed_text:
        answer = seed_text
        question_type = (suggested_question_type or "").strip() or "general"
        if suggested_candidate_question_types and len(suggested_candidate_question_types) > 0:
            candidate_question_types = list(suggested_candidate_question_types)
        else:
            candidate_question_types = [question_type]
        logger.info(
            "run_system 复用 Get Suggest 初答（跳过 student.answer）question_type=%s candidate_question_types=%s",
            question_type,
            candidate_question_types,
        )
    else:
        answer, question_type, candidate_question_types = student.answer(question, ltm, mk_data)
        logger.info(
            "[TRACE] run question_type=%s candidate_question_types=%s",
            question_type,
            candidate_question_types,
        )
    confidence, consistency = student.evaluate_answer(answer, ltm)
    logger.info(
        "初答完成 question_type=%s candidate_question_types=%s confidence=%.3f consistency=%.3f",
        question_type,
        candidate_question_types,
        confidence,
        consistency,
    )
    logger.info("初答内容(全文): %s", answer)
    initial_answer = answer
    loop_count = 0
    expert_knowledge_cache: dict[str, dict[str, Any]] = {}
    # 至少执行 1 轮多 Agent 反思，再由 should_continue 判定是否继续第 2 轮及后续轮次
    should_continue = True

    collected_expert_feedbacks = []
    collected_general_feedbacks = []

    while should_continue:
        general_feedbacks = []
        if selected_agents is not None and len(selected_agents) > 0:
            general_pool = [x for x in selected_agents if x in GeneralAgentFactory.all_names()]
            general_names = general_pool[:3] if general_pool else GeneralAgentFactory.sample_names(3)
            expert_agent = ExpertAgentFactory.create_by_question_type(question_type)
            expert_name = expert_agent.agent_name
            agents_this_round = general_names + [expert_name]
        else:
            agents_this_round = mk.select_agents(
                question, confidence, consistency, question_type=question_type
            )
        round_num = loop_count + 1
        logger.info(
            "[TRACE] round=%s general_agents=%s expert_agent=%s",
            round_num,
            agents_this_round[:3],
            agents_this_round[3] if len(agents_this_round) > 3 else "",
        )
        logger.info("========== 第 %s 轮 ========== 本轮 Agent: %s", round_num, agents_this_round)
        selected_general_agents = [
            agent_name for agent_name in agents_this_round if agent_name in GeneralAgentFactory.all_names()
        ]
        active_expert_bundle: dict[str, Any] | None = None
        expert_feedbacks: list[dict[str, Any]] = []
        for agent_name in agents_this_round:
            if agent_name in GeneralAgentFactory.all_names():
                continue
            expert_agent = ExpertAgentFactory.create(agent_name)
            knowledge_bundle = expert_knowledge_cache.get(agent_name)
            if knowledge_bundle is None:
                knowledge_bundle = expert_agent.retrieve_knowledge_bundle(
                    question,
                    candidate_question_types,
                )
                expert_knowledge_cache[agent_name] = knowledge_bundle
                logger.info(
                    "Expert 知识缓存写入 agent=%s exact_found=%s hit_count=%s search_types=%s",
                    agent_name,
                    knowledge_bundle.get("exact_found"),
                    knowledge_bundle.get("hit_count"),
                    knowledge_bundle.get("search_types"),
                )
            else:
                logger.info(
                    "Expert 知识缓存复用 agent=%s exact_found=%s hit_count=%s search_types=%s",
                    agent_name,
                    knowledge_bundle.get("exact_found"),
                    knowledge_bundle.get("hit_count"),
                    knowledge_bundle.get("search_types"),
                )
            active_expert_bundle = knowledge_bundle
            expert_feedbacks.append(
                expert_agent.review(
                    answer,
                    question,
                    candidate_question_types=candidate_question_types,
                    cached_knowledge_bundle=knowledge_bundle,
                )
            )

        if expert_feedbacks:
            _log_feedbacks(expert_feedbacks, prefix="  ")

        exact_found = bool(active_expert_bundle and active_expert_bundle.get("exact_found"))
        canonical_answer = (
            str(active_expert_bundle.get("canonical_answer") or "").strip()
            if active_expert_bundle
            else ""
        )

        if exact_found and expert_feedbacks:
            feedbacks = list(expert_feedbacks)
            all_experts_aligned = all(
                str(fb.get("expert_verdict") or "").strip().lower() == "aligned"
                for fb in expert_feedbacks
            )
            expert_guidance = _feedbacks_to_text(expert_feedbacks)
            reviewer_guidance = f"【阶段一 Expert 对齐】\n{expert_guidance}" if expert_guidance else ""
            logger.info("阶段一 Expert 对齐指导(全文): %s", expert_guidance)
            if all_experts_aligned:
                aligned_answer = answer
                logger.info("阶段一 Expert 判定当前答案已与标答一致，跳过第二阶段。")
            else:
                aligned_answer = student.revise_answer(
                    question,
                    answer,
                    expert_guidance,
                    stage="expert_alignment",
                    canonical_answer="",
                ).strip() or answer
            logger.info("阶段一 Expert 对齐结果(全文): %s", aligned_answer)

            if all_experts_aligned:
                candidate_answer = aligned_answer
            elif selected_general_agents:
                general_feedbacks = [
                    GeneralAgentFactory.create(agent_name).review(aligned_answer, question)
                    for agent_name in selected_general_agents
                ]
                logger.info("阶段二 General 重新审查基于阶段一答案")
                _log_feedbacks(general_feedbacks, prefix="  ")
                feedbacks.extend(general_feedbacks)
                general_guidance = insight.integrate_feedback(
                    question,
                    aligned_answer,
                    general_feedbacks,
                    polish_only=True,
                    expert_anchor=expert_guidance if expert_guidance else None,
                )
                if (general_guidance or "").strip():
                    reviewer_guidance = (
                        reviewer_guidance + f"\n\n【阶段二 General 润色】\n{general_guidance}"
                        if reviewer_guidance
                        else f"【阶段二 General 润色】\n{general_guidance}"
                    )
                    logger.info("阶段二 General 润色指导(全文): %s", general_guidance)
                    candidate_answer = student.revise_answer(
                        question,
                        aligned_answer,
                        general_guidance,
                        stage="general_polish",
                        canonical_answer=canonical_answer,
                    ).strip() or aligned_answer
                else:
                    logger.info("阶段二无安全润色点，跳过 Student 改写，保留 Expert 对齐答案")
                    candidate_answer = aligned_answer
            else:
                candidate_answer = aligned_answer
        else:
            general_feedbacks = [
                GeneralAgentFactory.create(agent_name).review(answer, question)
                for agent_name in selected_general_agents
            ]
            if general_feedbacks:
                _log_feedbacks(general_feedbacks, prefix="  ")
            feedbacks = expert_feedbacks + general_feedbacks
            reviewer_guidance = insight.integrate_feedback(question, answer, feedbacks)
            logger.info("Insight 综合指导(全文): %s", reviewer_guidance)
            candidate_answer = student.revise_answer(question, answer, reviewer_guidance)
        logger.info("本轮候选改写答案(全文): %s", candidate_answer)
        improved_answer, accept_meta = student.choose_better_answer(
            question,
            answer,
            candidate_answer,
            reviewer_guidance=reviewer_guidance,
        )
        improvement_score = compute_improvement(answer, improved_answer)
        accepted = bool(accept_meta.get("accepted"))
        logger.info(
            "本轮验收结果 accepted=%s reason=%s improvement_score=%.3f",
            accepted,
            accept_meta.get("reason", ""),
            improvement_score,
        )
        logger.info("本轮采纳后答案(全文): %s", improved_answer)
        if accepted:
            should_continue = mk.should_continue(
                answer,
                improved_answer,
                improvement_score,
                loop_count,
                initial_answer=initial_answer,
            )
        else:
            should_continue = False
            logger.info("should_continue 结束原因: 候选改写未通过验收，停止继续迭代")

        if expert_feedbacks:
            collected_expert_feedbacks.extend(expert_feedbacks)
        if general_feedbacks:
            collected_general_feedbacks.extend(general_feedbacks)

        update_wm(
            wm,
            question=question,
            student_answer=answer,
            agent_feedback=feedbacks,
            improved_answer=improved_answer,
            iteration=loop_count,
        )
        top_contributor_agents: list[str] = []
        rewarded_general_agents: list[str] = []
        if do_update_mk and selected_general_agents:
            top_contributor_agents = insight.select_major_contributors(
                question,
                answer,
                feedbacks,
                improved_answer,
                candidate_agents=selected_general_agents,
                accepted=accepted,
                improvement_score=improvement_score,
            )
            rewarded_general_agents = [
                agent_name for agent_name in top_contributor_agents if agent_name in selected_general_agents
            ]
            logger.info(
                "本轮贡献较大的 Agents=%s，获得奖励的通用 Agents=%s",
                top_contributor_agents,
                rewarded_general_agents,
            )
        if do_update_mk and selected_general_agents:
            evolve_mk_from_better_agents(
                mk_data,
                question_type,
                rewarded_general_agents,
                all_agent_names=selected_general_agents,
                reward_delta=1.0,
            )

        prev_answer_this_round = answer
        answer = improved_answer
        loop_count += 1
        logger.info("第 %s 轮结束，是否继续下一轮: %s", loop_count, should_continue)
        logger.info(
            "[TRACE] round=%s improvement=%.3f should_continue=%s",
            loop_count,
            improvement_score,
            should_continue,
        )
        confidence, consistency = student.evaluate_answer(
            answer, ltm, initial_answer=initial_answer, previous_answer=prev_answer_this_round
        )

    if do_update_mk:
        logger.info("更新并保存 MK")
        update_mk_from_ltm(mk_data, ltm)
        save_mk(mk_data)

    if do_update_ltm:
        logger.info("当前阶段已禁用 LTM 写回，跳过更新。")

    logger.info("run_system 完成 共 %s 轮，最终答案(全文): %s", loop_count, answer)
    return {
        "initial_answer": initial_answer,
        "final_answer": answer,
        "expert_feedbacks": collected_expert_feedbacks,
        "general_feedbacks": collected_general_feedbacks
    }


if __name__ == "__main__":
    import argparse
    import json
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    parser = argparse.ArgumentParser(description="批量运行多 Agent 反思系统")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集文件路径（JSON 数组，元素需包含 question 字段）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果输出文件路径（默认 testdata/results.json）",
    )
    args = parser.parse_args()

    default_dataset = Path(__file__).parent / "testdata" / "test.json"
    default_output = Path(__file__).parent / "testdata" / "results.json"
    test_path = Path(args.dataset).expanduser() if args.dataset else default_dataset
    out_path = Path(args.output).expanduser() if args.output else default_output

    with open(test_path, encoding="utf-8") as f:
        items = json.load(f)

    results: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        q = item.get("question", "")
        if not q:
            continue
        print(f"\n========== 第 {i + 1}/{len(items)} 题 ==========")
        print("问题:", q)
        reset_token_usage()
        t0 = time.perf_counter()
        run_result = run_system(q)
        elapsed = time.perf_counter() - t0
        usage = get_token_usage()
        record = {
            "问题": q,
            "第一次回答的答案": run_result["initial_answer"],
            "最终回答的答案": run_result["final_answer"],
            "回答时间": datetime.now(timezone.utc).isoformat(),
            "耗时_秒": round(elapsed, 2),
            "消耗的token": usage["total_tokens"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
        }
        results.append(record)
        print("=== 最终答案 ===")
        print(run_result["final_answer"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至 {out_path}")
