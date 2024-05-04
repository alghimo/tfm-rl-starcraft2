from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..actions import AllActions
from ..types import AgentStage, RewardMethod


@dataclass
class EpisodeStats:
    map_name: str
    reward: int = 0
    adjusted_reward: int = 0
    steps: int = 0
    epsilon: float = 1.
    losses: List[float] = field(default_factory=list)
    emissions: float = 0.
    score: int = 0
    is_burnin: bool = False
    is_training: bool = False
    is_exploit: bool = False
    is_random_mode: bool = False
    initial_stage: str = AgentStage.UNKNOWN.name
    final_stage: str = AgentStage.UNKNOWN.name
    episode: int = 0
    loss: float = 0.

    invalid_action_counts: Dict[str, int] = field(default_factory=lambda: {a.name: 0 for a in list(AllActions)})
    valid_action_counts: Dict[str, int] = field(default_factory=lambda: {a.name: 0 for a in list(AllActions)})

    @property
    def mean_loss(self) -> float:
        if any(self.losses):
            return sum(self.losses) / len(self.losses)
        return np.inf

    def add_invalid_action(self, action: AllActions):
        self.invalid_action_counts[action.name] += 1

    def add_valid_action(self, action: AllActions):
        self.valid_action_counts[action.name] += 1

@dataclass
class AgentStats:
    map_name: str
    step_count: int = 0
    episode_count: int = 0

    total_reward: int = 0
    total_adjusted_reward: int = 0
    total_score: int = 0
    total_emissions: float = 0.
    invalid_action_counts: Dict[str, int] = field(default_factory=lambda: {a.name: 0 for a in list(AllActions)})
    valid_action_counts: Dict[str, int] = field(default_factory=lambda: {a.name: 0 for a in list(AllActions)})

    step_count_per_stage: Dict[str, int] = field(default_factory=lambda: {s.name: 0 for s in list(AgentStage)})
    episode_count_per_stage: Dict[str, int] = field(default_factory=lambda: {s.name: 0 for s in list(AgentStage)})
    total_reward_per_stage: Dict[str, int] = field(default_factory=lambda: {s.name: 0 for s in list(AgentStage)})
    total_adjusted_reward_per_stage: Dict[str, int] = field(default_factory=lambda: {s.name: 0 for s in list(AgentStage)})
    total_score_per_stage: Dict[str, int] = field(default_factory=lambda: {s.name: 0 for s in list(AgentStage)})
    total_emissions_per_stage: Dict[str, int] = field(default_factory=lambda: {s.name: 0 for s in list(AgentStage)})
    invalid_action_counts_per_stage: Dict[str, int] = field(default_factory=lambda: {s.name: {a.name: 0 for a in list(AllActions)} for s in list(AgentStage)})
    valid_action_counts_per_stage: Dict[str, int] = field(default_factory=lambda: {s.name: {a.name: 0 for a in list(AllActions)} for s in list(AgentStage)})

    def process_episode(self, episode_stats: EpisodeStats):
        self.total_emissions += episode_stats.emissions
        self.total_reward += episode_stats.reward
        self.total_adjusted_reward += episode_stats.adjusted_reward
        self.total_score += episode_stats.score
        for a, c in episode_stats.invalid_action_counts.items():
            self.invalid_action_counts[a] += c

        for a, c in episode_stats.valid_action_counts.items():
            self.valid_action_counts[a] += c

        stage = episode_stats.initial_stage

        self.total_emissions_per_stage[stage] += episode_stats.emissions
        self.total_reward_per_stage[stage] += episode_stats.reward
        self.total_adjusted_reward_per_stage[stage] += episode_stats.adjusted_reward
        self.total_score_per_stage[stage] += episode_stats.score
        for a, c in episode_stats.invalid_action_counts.items():
            self.invalid_action_counts_per_stage[stage][a] += c

        for a, c in episode_stats.valid_action_counts.items():
            self.valid_action_counts_per_stage[stage][a] += c


@dataclass
class AggregatedEpisodeStats:
    map_name: str
    rewards: List[int] = field(default_factory=list)
    adjusted_rewards: List[int] = field(default_factory=list)
    train_rewards: List[int] = field(default_factory=list)
    train_adjusted_rewards: List[int] = field(default_factory=list)
    train_scores: List[int] = field(default_factory=list)
    train_steps: List[int] = field(default_factory=list)
    scores: List[int] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    emissions: List[float] = field(default_factory=list)
    max_reward: float = None
    max_adjusted_reward: float = None
    max_score: float = None
    invalid_action_counts: Dict[str, int] = field(default_factory=lambda: {a.name: 0 for a in list(AllActions)})
    valid_action_counts: Dict[str, int] = field(default_factory=lambda: {a.name: 0 for a in list(AllActions)})

    rewards_per_stage: Dict[str, List[float]] = field(default_factory=lambda: {s.name: [] for s in list(AgentStage)})
    adjusted_rewards_per_stage: Dict[str, List[float]] = field(default_factory=lambda: {s.name: [] for s in list(AgentStage)})
    scores_per_stage: Dict[str, List[float]] = field(default_factory=lambda: {s.name: [] for s in list(AgentStage)})
    epsilons_per_stage: Dict[str, List[float]] = field(default_factory=lambda: {s.name: [] for s in list(AgentStage)})
    losses_per_stage: Dict[str, List[float]] = field(default_factory=lambda: {s.name: [] for s in list(AgentStage)})
    steps_per_stage: Dict[str, List[int]] = field(default_factory=lambda: {s.name: [] for s in list(AgentStage)})
    emissions_per_stage: Dict[str, List[float]] = field(default_factory=lambda: {s.name: [] for s in list(AgentStage)})
    max_reward_per_stage: Dict[str, float] = field(default_factory=lambda: {s.name: None for s in list(AgentStage)})
    max_adjusted_reward_per_stage: Dict[str, float] = field(default_factory=lambda: {s.name: None for s in list(AgentStage)})
    max_score_per_stage: Dict[str, float] = field(default_factory=lambda: {s.name: None for s in list(AgentStage)})
    invalid_action_counts_per_stage: Dict[str, int] = field(default_factory=lambda: {s.name: {a.name: 0 for a in list(AllActions)} for s in list(AgentStage)})
    valid_action_counts_per_stage: Dict[str, int] = field(default_factory=lambda: {s.name: {a.name: 0 for a in list(AllActions)} for s in list(AgentStage)})

    @property
    def mean_emissions(self) -> float:
        return np.mean(self.emissions)

    def mean_emissions_last_n_episodes(self, n: int = 10) -> float:
        return np.mean(self.emissions[-n:])

    def mean_rewards(self, stage: AgentStage = None, last_n: int = None, reward_method: RewardMethod = RewardMethod.REWARD) -> float:
        if reward_method == RewardMethod.REWARD:
            rewards_source = self.rewards if stage is None else self.rewards_per_stage[stage.name]
        elif reward_method == RewardMethod.ADJUSTED_REWARD:
            rewards_source = self.adjusted_rewards if stage is None else self.adjusted_rewards_per_stage[stage.name]
        elif reward_method == RewardMethod.SCORE:
            rewards_source = self.scores if stage is None else self.scores_per_stage[stage.name]

        if last_n is None:
            return np.mean(rewards_source)
        else:
            return np.mean(rewards_source[-last_n:])

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
        self.adjusted_rewards.append(episode_stats.adjusted_reward)
        self.scores.append(episode_stats.score)
        self.steps.append(episode_stats.steps)
        self.epsilons.append(episode_stats.epsilon)
        self.emissions.append(episode_stats.emissions)
        self.losses.append(episode_stats.mean_loss)

        if self.max_reward is None or self.max_reward < episode_stats.reward:
            self.max_reward = episode_stats.reward

        if self.max_adjusted_reward is None or self.max_adjusted_reward < episode_stats.adjusted_reward:
            self.max_adjusted_reward = episode_stats.adjusted_reward

        if self.max_score is None or self.max_score < episode_stats.score:
            self.max_score = episode_stats.score

        for a, c in episode_stats.invalid_action_counts.items():
            self.invalid_action_counts[a] += c

        for a, c in episode_stats.valid_action_counts.items():
            self.valid_action_counts[a] += c

        stage = episode_stats.initial_stage

        self.rewards_per_stage[stage].append(episode_stats.reward)
        self.adjusted_rewards_per_stage[stage].append(episode_stats.adjusted_reward)
        self.scores_per_stage[stage].append(episode_stats.score)
        self.epsilons_per_stage[stage].append(episode_stats.epsilon)
        self.losses_per_stage[stage].append(episode_stats.mean_loss)
        self.steps_per_stage[stage].append(episode_stats.steps)
        self.emissions_per_stage[stage].append(episode_stats.emissions)

        if self.max_reward_per_stage[stage] is None or self.max_reward_per_stage[stage] < episode_stats.reward:
            self.max_reward_per_stage[stage] = episode_stats.reward

        if self.max_adjusted_reward_per_stage[stage] is None or self.max_adjusted_reward_per_stage[stage] < episode_stats.adjusted_reward:
            self.max_adjusted_reward_per_stage[stage] = episode_stats.adjusted_reward

        if self.max_score_per_stage[stage] is None or self.max_score_per_stage[stage] < episode_stats.score:
            self.max_score_per_stage[stage] = episode_stats.score

        for a, c in episode_stats.invalid_action_counts.items():
            self.invalid_action_counts_per_stage[stage][a] += c

        for a, c in episode_stats.valid_action_counts.items():
            self.valid_action_counts_per_stage[stage][a] += c
