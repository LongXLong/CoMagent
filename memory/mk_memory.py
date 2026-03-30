# -*- coding: utf-8 -*-
"""MK Memory：固定 16 类策略配置。"""

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import DEFAULT_QUESTION_TYPE, MK_MEMORY_PATH
from memory.ltm import QUESTION_TYPES_16, infer_question_type_for_ltm
from utils.logger import get_logger

logger = get_logger(__name__)


def _default_type() -> str:
    return "OTHER_GENERAL_Q"


def _default_type_config() -> dict[str, Any]:
    return {
        "description": "固定 16 类策略",
        "strategy": {
            "similarity_threshold": 0.9,
            "improvement_min": 0.05,
            "max_loops": 3,
        },
        "agent_priorities": {},
        "thresholds": {
            "confidence_low": 0.6,
            "consistency_low": 0.7,
            "confidence_high": 0.8,
            "consistency_high": 0.8,
        },
        "agent_effectiveness": {},
        "agent_mab_stats": {},
    }


def _default_mk() -> dict[str, Any]:
    return {
        "question_types": {qt: copy.deepcopy(_default_type_config()) for qt in QUESTION_TYPES_16},
        "default_type": _default_type(),
        "last_update": "",
    }


def load_mk(path: Path | None = None) -> dict[str, Any]:
    p = path or MK_MEMORY_PATH
    if not p.exists():
        return _default_mk()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "question_types" not in data:
        return _default_mk()
    out = _default_mk()
    in_types = data.get("question_types", {})
    if isinstance(in_types, dict):
        for qt in QUESTION_TYPES_16:
            cfg = in_types.get(qt)
            if isinstance(cfg, dict):
                out["question_types"][qt].update(cfg)
    out["default_type"] = data.get("default_type", _default_type())
    out["last_update"] = data.get("last_update", "")
    return out


def save_mk(mk: dict[str, Any], path: Path | None = None) -> None:
    p = path or MK_MEMORY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(mk, f, ensure_ascii=False, indent=2)


def infer_question_type(question: str, mk: dict[str, Any] | None = None) -> str:
    qt = infer_question_type_for_ltm(question)
    if qt in QUESTION_TYPES_16:
        return qt
    return (mk or {}).get("default_type", _default_type())


def get_config_for_question_type(mk: dict[str, Any], question_type: str) -> dict[str, Any]:
    types = mk.get("question_types", {})
    default_type = mk.get("default_type", _default_type())
    qt = question_type if question_type in types else default_type
    if qt not in types:
        qt = _default_type()
    t = types.get(qt, _default_type_config())
    return {
        "strategy": t.get("strategy", {}),
        "agent_priorities": t.get("agent_priorities", {}),
        "thresholds": t.get("thresholds", {}),
        "agent_effectiveness": t.get("agent_effectiveness", {}),
        "agent_mab_stats": t.get("agent_mab_stats", {}),
    }


def update_mk_from_ltm(mk: dict[str, Any], ltm: dict[str, Any]) -> None:
    mk.setdefault("question_types", {})
    for qt in QUESTION_TYPES_16:
        mk["question_types"].setdefault(qt, copy.deepcopy(_default_type_config()))
    mk["default_type"] = mk.get("default_type", _default_type())
    mk["last_update"] = datetime.now().strftime("%Y-%m-%d")


def _ensure_agent_mab_stats(
    type_config: dict[str, Any],
    agent_names: list[str],
) -> dict[str, Any]:
    stats = type_config.setdefault("agent_mab_stats", {})
    for agent_name in agent_names:
        stat = stats.setdefault(agent_name, {})
        stat.setdefault("reward_sum", 0.0)
        stat.setdefault("sq_reward_sum", 0.0)
        stat.setdefault("total_tasks", 0)
    return stats


def update_agent_mab_stats(
    mk: dict[str, Any],
    question_type: str,
    selected_agent_names: list[str],
    *,
    all_agent_names: list[str] | None = None,
    reward_delta: float = 0.0,
    sq_reward_delta: float | None = None,
    unselected_reward_delta: float = 0.0,
    unselected_sq_reward_delta: float = 0.0,
    task_delta: int = 1,
) -> None:
    types = mk.get("question_types", {})
    if question_type not in types:
        return

    selected = {name for name in selected_agent_names if name}
    tracked_agents = list(dict.fromkeys((all_agent_names or []) + list(selected)))
    if not tracked_agents:
        return

    type_config = types[question_type]
    stats = _ensure_agent_mab_stats(type_config, tracked_agents)
    selected_sq_reward_delta = reward_delta * reward_delta if sq_reward_delta is None else sq_reward_delta

    for agent_name in tracked_agents:
        stat = stats[agent_name]
        is_selected = agent_name in selected
        stat["reward_sum"] = float(stat.get("reward_sum", 0.0)) + (
            reward_delta if is_selected else unselected_reward_delta
        )
        stat["sq_reward_sum"] = float(stat.get("sq_reward_sum", 0.0)) + (
            selected_sq_reward_delta if is_selected else unselected_sq_reward_delta
        )
        stat["total_tasks"] = int(stat.get("total_tasks", 0)) + int(task_delta)


def evolve_mk_from_random_agent(
    mk: dict[str, Any],
    question_type: str,
    random_agent_name: str,
    *,
    effectiveness_delta: float = 0.02,
    priority_delta: float = 0.02,
    all_agent_names: list[str] | None = None,
    reward_delta: float | None = None,
    sq_reward_delta: float | None = None,
) -> None:
    types = mk.get("question_types", {})
    if question_type not in types:
        return
    t = types[question_type]
    eff = t.setdefault("agent_effectiveness", {})
    prio = t.setdefault("agent_priorities", {})
    eff[random_agent_name] = min(1.0, float(eff.get(random_agent_name, 0.5)) + effectiveness_delta)
    prio[random_agent_name] = min(1.0, float(prio.get(random_agent_name, 0.5)) + priority_delta)
    update_agent_mab_stats(
        mk,
        question_type,
        [random_agent_name],
        all_agent_names=all_agent_names,
        reward_delta=effectiveness_delta if reward_delta is None else reward_delta,
        sq_reward_delta=sq_reward_delta,
    )
    mk["last_update"] = datetime.now().strftime("%Y-%m-%d")


def select_better_agents_from_wm(
    wm: dict[str, Any],
    final_answer: str,
    *,
    max_agents: int = 3,
) -> list[str]:
    history = wm.get("agent_feedback_history", [])
    if not history:
        return []
    scores: dict[str, int] = {}
    for round_feedback in history:
        for fb in round_feedback:
            name = str(fb.get("agent", "")).strip()
            comment = str(fb.get("comment", "")).strip()
            if name and comment:
                scores[name] = scores.get(name, 0) + 1
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:max_agents]]


def evolve_mk_from_better_agents(
    mk: dict[str, Any],
    question_type: str,
    better_agent_names: list[str],
    *,
    effectiveness_delta: float = 0.03,
    priority_delta: float = 0.03,
    all_agent_names: list[str] | None = None,
    reward_delta: float | None = None,
    sq_reward_delta: float | None = None,
) -> None:
    types = mk.get("question_types", {})
    if question_type not in types:
        return
    t = types[question_type]
    eff = t.setdefault("agent_effectiveness", {})
    prio = t.setdefault("agent_priorities", {})
    for name in better_agent_names:
        eff[name] = min(1.0, float(eff.get(name, 0.5)) + effectiveness_delta)
        prio[name] = min(1.0, float(prio.get(name, 0.5)) + priority_delta)
    update_agent_mab_stats(
        mk,
        question_type,
        better_agent_names,
        all_agent_names=all_agent_names,
        reward_delta=effectiveness_delta if reward_delta is None else reward_delta,
        sq_reward_delta=sq_reward_delta,
    )
    mk["last_update"] = datetime.now().strftime("%Y-%m-%d")
