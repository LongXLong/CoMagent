# -*- coding: utf-8 -*-
"""
Multi-armed bandit selection utilities for general agents.

Supported algorithms:
- UCB
- UCBTuned
- ThompsonSampling
- Random
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence


# Current bandit algorithm: "UCB" | "UCBTuned" | "ThompsonSampling" | "Random"
MAB_ALGORITHM = "ThompsonSampling"


def _available_arms(n_arms: int, masked: frozenset[int] | None = None) -> list[int]:
    if masked is None or not masked:
        return list(range(n_arms))
    return [i for i in range(n_arms) if i not in masked]


class BaseBandit:
    def __init__(self, n_arms: int):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        self.n_arms = n_arms
        self.total_steps = 0

    def select(self, masked: frozenset[int] | None = None) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        raise NotImplementedError

    def get_scores(self) -> list[float] | None:
        return None

    def set_warm_start(
        self,
        counts: list[int],
        sum_rewards: list[float],
        sum_sq_rewards: list[float] | None = None,
    ) -> None:
        return


class UCB1Bandit(BaseBandit):
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.counts = [0] * n_arms
        self.sums = [0.0] * n_arms

    def select(self, masked: frozenset[int] | None = None) -> int:
        self.total_steps += 1
        available = _available_arms(self.n_arms, masked)
        cold_arms = [arm for arm in available if self.counts[arm] == 0]
        if cold_arms:
            return random.choice(cold_arms)

        log_term = math.log(max(1, self.total_steps))
        scores = []
        for arm in range(self.n_arms):
            mean = self.sums[arm] / max(1, self.counts[arm])
            bonus = math.sqrt(2.0 * log_term / max(1, self.counts[arm]))
            scores.append(mean + bonus)
        return max(available, key=lambda arm: scores[arm])

    def update(self, arm: int, reward: float) -> None:
        reward = float(min(1.0, max(0.0, reward)))
        self.counts[arm] += 1
        self.sums[arm] += reward

    def get_scores(self) -> list[float]:
        log_term = math.log(max(1, self.total_steps))
        scores: list[float] = []
        for arm in range(self.n_arms):
            mean = self.sums[arm] / max(1, self.counts[arm])
            bonus = math.sqrt(2.0 * log_term / max(1, self.counts[arm]))
            scores.append(mean + bonus)
        return scores

    def set_warm_start(
        self,
        counts: list[int],
        sum_rewards: list[float],
        sum_sq_rewards: list[float] | None = None,
    ) -> None:
        self.counts = list(counts)
        self.sums = list(sum_rewards)
        self.total_steps = max(1, sum(self.counts))


class UCBTunedBandit(BaseBandit):
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.counts = [0] * n_arms
        self.sums = [0.0] * n_arms
        self.sq_sums = [0.0] * n_arms

    def select(self, masked: frozenset[int] | None = None) -> int:
        self.total_steps += 1
        available = _available_arms(self.n_arms, masked)
        cold_arms = [arm for arm in available if self.counts[arm] == 0]
        if cold_arms:
            return random.choice(cold_arms)

        log_term = math.log(max(1, self.total_steps))
        scores: list[float] = []
        for arm in range(self.n_arms):
            count = max(1, self.counts[arm])
            mean = self.sums[arm] / count
            variance = max(0.0, (self.sq_sums[arm] / count) - mean * mean)
            variance_bound = min(0.25, variance + math.sqrt(2.0 * log_term / count))
            bonus = math.sqrt((log_term / count) * variance_bound)
            scores.append(mean + bonus)
        return max(available, key=lambda arm: scores[arm])

    def update(self, arm: int, reward: float) -> None:
        reward = float(min(1.0, max(0.0, reward)))
        self.counts[arm] += 1
        self.sums[arm] += reward
        self.sq_sums[arm] += reward * reward

    def get_scores(self) -> list[float]:
        log_term = math.log(max(1, self.total_steps))
        scores: list[float] = []
        for arm in range(self.n_arms):
            count = max(1, self.counts[arm])
            mean = self.sums[arm] / count
            variance = max(0.0, (self.sq_sums[arm] / count) - mean * mean)
            variance_bound = min(0.25, variance + math.sqrt(2.0 * log_term / count))
            bonus = math.sqrt((log_term / count) * variance_bound)
            scores.append(mean + bonus)
        return scores

    def set_warm_start(
        self,
        counts: list[int],
        sum_rewards: list[float],
        sum_sq_rewards: list[float] | None = None,
    ) -> None:
        sq_rewards = sum_sq_rewards if sum_sq_rewards is not None else [r * r for r in sum_rewards]
        self.counts = list(counts)
        self.sums = list(sum_rewards)
        self.sq_sums = list(sq_rewards)
        self.total_steps = max(1, sum(self.counts))


class ThompsonSamplingBandit(BaseBandit):
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.alphas = [1.0] * n_arms
        self.betas = [1.0] * n_arms
        self._last_samples: list[float] | None = None

    def select(self, masked: frozenset[int] | None = None) -> int:
        self.total_steps += 1
        available = _available_arms(self.n_arms, masked)
        samples = {arm: random.betavariate(self.alphas[arm], self.betas[arm]) for arm in available}
        self._last_samples = [samples.get(i, 0.0) for i in range(self.n_arms)]
        return max(available, key=lambda arm: samples[arm])

    def select_top_k(self, k: int) -> list[int]:
        self.total_steps += 1
        k = max(1, min(k, self.n_arms))
        samples = [random.betavariate(self.alphas[i], self.betas[i]) for i in range(self.n_arms)]
        self._last_samples = samples
        ranked = sorted(range(self.n_arms), key=lambda idx: samples[idx], reverse=True)
        return ranked[:k]

    def update(self, arm: int, reward: float) -> None:
        reward = float(min(1.0, max(0.0, reward)))
        self.alphas[arm] += reward
        self.betas[arm] += 1.0 - reward

    def get_scores(self) -> list[float]:
        return [
            self.alphas[i] / max(1e-9, self.alphas[i] + self.betas[i])
            for i in range(self.n_arms)
        ]

    def set_warm_start(
        self,
        counts: list[int],
        sum_rewards: list[float],
        sum_sq_rewards: list[float] | None = None,
    ) -> None:
        for idx in range(self.n_arms):
            count = counts[idx] if idx < len(counts) else 0
            reward_sum = sum_rewards[idx] if idx < len(sum_rewards) else 0.0
            self.alphas[idx] = 1.0 + reward_sum
            self.betas[idx] = 1.0 + max(0.0, count - reward_sum)
        self.total_steps = max(1, sum(counts))


class RandomBandit(BaseBandit):
    def select(self, masked: frozenset[int] | None = None) -> int:
        self.total_steps += 1
        return random.choice(_available_arms(self.n_arms, masked))

    def update(self, arm: int, reward: float) -> None:
        return

    def get_scores(self) -> list[float]:
        return [0.0] * self.n_arms


def create_bandit(algo: str, n_arms: int) -> BaseBandit:
    normalized = (algo or "").strip().upper().replace("-", "").replace("_", "")
    if normalized in {"UCB", "UCB1"}:
        return UCB1Bandit(n_arms)
    if normalized == "UCBTUNED":
        return UCBTunedBandit(n_arms)
    if normalized in {"THOMPSONSAMPLING", "TS"}:
        return ThompsonSamplingBandit(n_arms)
    if normalized == "RANDOM":
        return RandomBandit(n_arms)
    return UCB1Bandit(n_arms)


class AgentBanditManager:
    def __init__(self, agent_ids: Sequence[str], algo: str | None = None):
        if not agent_ids:
            raise ValueError("agent_ids cannot be empty")
        self.agent_ids = list(agent_ids)
        self.id_to_index = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}
        self.index_to_id = {idx: agent_id for idx, agent_id in enumerate(self.agent_ids)}
        self.algo = (algo or MAB_ALGORITHM).strip() or "UCB"
        self.bandit = create_bandit(self.algo, len(self.agent_ids))

    def warm_start_from_mab_stats(self, agent_mab_stats: dict[str, Any]) -> None:
        counts: list[int] = []
        reward_sums: list[float] = []
        sq_reward_sums: list[float] = []
        for agent_id in self.agent_ids:
            stat = agent_mab_stats.get(agent_id) or {}
            counts.append(int(stat.get("total_tasks", 0) or 0))
            reward_sums.append(float(stat.get("reward_sum", 0.0) or 0.0))
            sq_reward_sums.append(float(stat.get("sq_reward_sum", 0.0) or 0.0))
        self.bandit.set_warm_start(counts, reward_sums, sq_reward_sums)

    def select_agents(self, k: int) -> list[str]:
        k = max(1, min(k, len(self.agent_ids)))
        if hasattr(self.bandit, "select_top_k"):
            selected_indices = self.bandit.select_top_k(k)
            return [self.index_to_id[idx] for idx in selected_indices]

        selected_indices: list[int] = []
        masked = frozenset()
        for _ in range(k):
            idx = self.bandit.select(masked=masked)
            selected_indices.append(idx)
            masked = frozenset(selected_indices)
        return [self.index_to_id[idx] for idx in selected_indices]

    def get_scores_by_agent(self) -> dict[str, float] | None:
        scores = self.bandit.get_scores()
        if scores is None or len(scores) != len(self.agent_ids):
            return None
        return {agent_id: float(score) for agent_id, score in zip(self.agent_ids, scores)}


def select_agents_by_mab(
    agent_ids: Sequence[str],
    agent_mab_stats: dict[str, Any] | None,
    *,
    k: int = 1,
    algo: str | None = None,
) -> tuple[list[str], dict[str, float] | None]:
    manager = AgentBanditManager(agent_ids, algo=algo)
    manager.warm_start_from_mab_stats(agent_mab_stats or {})
    selected = manager.select_agents(k=k)
    return selected, manager.get_scores_by_agent()
