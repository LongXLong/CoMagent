# -*- coding: utf-8 -*-
"""
基于 Judge LLM 的评估脚本（增强版）：
- 保留实时输出（准确率、耗时、token）
- 保留“模型答案 / 标准答案”输出
- 同时写入日志 log/app.log
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from config import JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL
from main import run_system
from utils.llm import get_token_usage, parse_json_from_llm, reset_token_usage

# 加载你的日志模块
from utils.logger import setup_logging, get_logger

# 初始化日志
setup_logging()
logger = get_logger("evaluator")


def _get_judge_client() -> OpenAI:
    if not JUDGE_API_KEY:
        logger.error("未设置 JUDGE_API_KEY！")
        raise ValueError("请设置 Judge API KEY")
    return OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    logger.info(f"加载数据集：{path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        logger.error("数据集格式错误")
        raise ValueError("数据集必须是 JSON 数组")
    logger.info(f"数据集加载成功，共 {len(data)} 条")
    return data


def _judge_consistency(question, reference_answer, model_answer, judge_model, judge_usage):
    """调用 Judge LLM 判断一致性"""
    system_prompt = "You are a strict answer consistency judge. Return JSON only."
    prompt = f"""
Question:
{question}

Reference Answer:
{reference_answer}

Model Answer:
{model_answer}

Task:
Determine whether Model Answer is consistent with Reference Answer.
Return:
{{"consistent": true/false}}
""".strip()

    client = _get_judge_client()
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    usage = getattr(resp, "usage", None)
    if usage:
        judge_usage["prompt_tokens"] += usage.prompt_tokens
        judge_usage["completion_tokens"] += usage.completion_tokens
        judge_usage["total_tokens"] += usage.total_tokens

    raw = (resp.choices[0].message.content or "").strip()
    obj = parse_json_from_llm(raw)
    if isinstance(obj, dict) and isinstance(obj.get("consistent"), bool):
        return obj["consistent"]

    # fallback
    lower = raw.lower()
    if "true" in lower:
        return True
    if "false" in lower:
        return False
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Judge LLM 一致性评估")
    parser.add_argument("--dataset", type=str, default="dataset\GSM8K.json")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    data = _load_dataset(dataset_path)

    if args.limit > 0:
        data = data[: args.limit]
        logger.info(f"限制评估前 {args.limit} 条")

    total = len(data)
    if total == 0:
        print("数据集为空。")
        logger.warning("数据集为空。")
        return

    reset_token_usage()
    judge_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    correct = 0
    skipped = 0
    processed = 0
    t_all_start = time.perf_counter()

    def _progress(done: int, total: int, width=28):
        ratio = max(0.0, min(1.0, done / total))
        filled = int(width * ratio)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    model_answer = ""
    reference = ""

    for i, item in enumerate(data, start=1):
        question = str(item.get("question", "")).strip()
        reference = str(item.get("answer", "")).strip()

        if not question or not reference:
            skipped += 1
            processed += 1
            continue

        t0 = time.perf_counter()

        try:
            result = run_system(
                question,
                selected_agents=None,
                do_update_ltm=True,
                do_update_mk=False,
            )
            model_answer = str(result.get("final_answer", "")).strip()

            is_correct = _judge_consistency(
                question, reference, model_answer, args.judge_model, judge_usage
            )

        except Exception as e:
            logger.error(f"第 {i} 条评估时出错：{e}")
            is_correct = False

        if is_correct:
            correct += 1
            logger.info(f"第 {i}/{total} 条：正确")
        else:
            logger.info(f"第 {i}/{total} 条：错误")

        processed += 1

        # --- 保留你的实时输出 ---
        elapsed = time.perf_counter() - t_all_start
        usage = get_token_usage()
        bar = _progress(processed, total)

        print(
            f"\r{bar} {processed}/{total} | "
            f"当前正确率: {(correct / max(1, processed - skipped)) * 100:.2f}% | "
            f"当前耗时: {elapsed:.2f}s | "
            f"当前消耗token: {usage.get('total_tokens', 0)}\n",
            end="",
            flush=True,
        )

    # 换行
    print()
    print(f"模型答案: {model_answer}")
    print(f"标准答案: {reference}")

    # --- 最终统计 ---
    total_elapsed = time.perf_counter() - t_all_start
    usage = get_token_usage()
    effective_total = total - skipped
    accuracy = (correct / max(1, effective_total)) * 100

    # 日志记录
    logger.info(f"最终正确率: {accuracy:.2f}%")
    logger.info(f"总耗时: {total_elapsed:.2f}s")
    logger.info(f"总 tokens: {usage.get('total_tokens', 0)}")

    # --- 保留你的原始输出 ---
    print()
    print(f"最终正确率: {accuracy:.2f}%")
    print(f"总统计耗时: {total_elapsed:.2f}s")
    print(f"总消耗token: {usage.get('total_tokens', 0)}")


if __name__ == "__main__":
    main()