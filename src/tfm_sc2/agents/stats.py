from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class AgentStats:
    map_name: str
    step_count: int = 0
    episode_count: int = 0


@dataclass
class EpisodeStats:
    map_name: str
    reward: int = 0
    steps: int = 0
    epsilon: float = 1.
    losses: List[float] = []

    @property
    def mean_loss(self) -> float:
        if any(self.losses):
            return sum(self.__current_episode_losses) / len(self.__current_episode_losses)
        return np.inf

@dataclass
class AggregatedEpisodeStats:
    map_name: str
    rewards: List[int] = []
    epsilon: List[float] = []
    losses: List[float] = []
    steps: List[int] = []
    max_reward: int = None

    @property
    def mean_rewards(self) -> float:
        return np.mean(self.rewards)

    def mean_rewards_last_n_episodes(self, n: int = 10) -> float:
        return np.mean(self.rewards[-n:])

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
        self.steps.append(episode_stats.steps)
        self.epsilons.append(episode_stats.epsilon)
        self.losses.append(episode_stats.mean_loss)

        if self.max_reward is None or self.max_reward < episode_stats.reward:
            self.max_reward = episode_stats.reward
