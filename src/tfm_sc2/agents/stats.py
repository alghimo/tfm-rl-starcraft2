from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class EpisodeStats:
    map_name: str
    reward: int = 0
    steps: int = 0
    epsilon: float = 1.
    losses: List[float] = field(default_factory=list)
    emissions: float = 0.
    score: int = 0

    @property
    def mean_loss(self) -> float:
        if any(self.losses):
            return sum(self.losses) / len(self.losses)
        return np.inf

@dataclass
class AgentStats:
    map_name: str
    step_count: int = 0
    episode_count: int = 0
    total_reward: int = 0
    total_score: int = 0
    total_emissions: float = 0.

    def process_episode(self, episode_stats: EpisodeStats):
        self.total_emissions += episode_stats.emissions
        self.total_reward += episode_stats.reward
        self.total_score += episode_stats.score

@dataclass
class AggregatedEpisodeStats:
    map_name: str
    rewards: List[int] = field(default_factory=list)
    scores: List[int] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    emissions: List[float] = field(default_factory=list)
    max_reward: int = None
    max_score: int = None

    @property
    def mean_emissions(self) -> float:
        return np.mean(self.emissions)

    def mean_emissions_last_n_episodes(self, n: int = 10) -> float:
        return np.mean(self.emissions[-n:])

    @property
    def mean_rewards(self) -> float:
        return np.mean(self.rewards)

    def mean_rewards_last_n_episodes(self, n: int = 10) -> float:
        return np.mean(self.rewards[-n:])

    @property
    def mean_score(self) -> float:
        return np.mean(self.scores)

    def mean_score_last_n_episodes(self, n: int = 10) -> float:
        return np.mean(self.scores[-n:])

    @property
    def mean_steps(self) -> float:
        return np.mean(self.steps)

    def mean_steps_last_last_n_episodes(self, n: int = 10) -> float:
        return np.mean(self.steps[-n:])

    @property
    def mean_losses(self) -> float:
        return np.mean(self.losses)

    def mean_losses_last_last_n_episodes(self, n: int = 10) -> float:
        return np.mean(self.losses[-n:])

    def process_episode(self, episode_stats: EpisodeStats):
        self.rewards.append(episode_stats.reward)
        self.scores.append(episode_stats.score)
        self.steps.append(episode_stats.steps)
        self.epsilons.append(episode_stats.epsilon)
        self.losses.append(episode_stats.mean_loss)
        self.emissions.append(episode_stats.emissions)

        if self.max_reward is None or self.max_reward < episode_stats.reward:
            self.max_reward = episode_stats.reward

        if self.max_score is None or self.max_score < episode_stats.score:
            self.max_score = episode_stats.score
