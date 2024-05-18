import logging
import pickle
import random
import time
from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from codecarbon.emissions_tracker import BaseEmissionsTracker
from pysc2.agents import base_agent
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, features, units
from pysc2.lib.features import PlayerRelative
from pysc2.lib.named_array import NamedNumpyArray
from typing_extensions import Self

from ..actions import AllActions
from ..constants import Constants, SC2Costs
from ..networks.experience_replay_buffer import ExperienceReplayBuffer
from ..types import AgentStage, Gas, Minerals, Position, RewardMethod, State
from ..with_logger import WithLogger
from .stats import AgentStats, AggregatedEpisodeStats, EpisodeStats


class BaseAgent(WithLogger, ABC, base_agent.BaseAgent):
    _AGENT_FILE: str = "agent.pkl"
    _BUFFER_FILE: str = "buffer.pkl"
    _STATS_FILE: str =  "stats.parquet"

    _action_to_game = {
        AllActions.NO_OP: actions.RAW_FUNCTIONS.no_op,
        AllActions.HARVEST_MINERALS: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", source_unit_tag, target_unit_tag),
        AllActions.RECRUIT_SCV: lambda source_unit_tag: actions.RAW_FUNCTIONS.Train_SCV_quick("now", source_unit_tag),
        AllActions.BUILD_SUPPLY_DEPOT: lambda source_unit_tag, target_position: actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", source_unit_tag, target_position),
        AllActions.BUILD_COMMAND_CENTER: lambda source_unit_tag, target_position: actions.RAW_FUNCTIONS.Build_CommandCenter_pt("now", source_unit_tag, target_position),
        AllActions.BUILD_BARRACKS: lambda source_unit_tag, target_position: actions.RAW_FUNCTIONS.Build_Barracks_pt("now", source_unit_tag, target_position),
        AllActions.RECRUIT_MARINE: lambda source_unit_tag: actions.RAW_FUNCTIONS.Train_Marine_quick("now", source_unit_tag),
        AllActions.ATTACK_WITH_SINGLE_UNIT: lambda source_unit_tag, target_position: actions.RAW_FUNCTIONS.Attack_pt("now", source_unit_tag, target_position),
    }

    def __init__(self,
                 map_name: str, map_config: Dict, buffer: ExperienceReplayBuffer = None,
                 train: bool = True, checkpoint_path: Union[str|Path] = None, tracker_update_freq_seconds: int = 10,
                 reward_method: RewardMethod = RewardMethod.REWARD, **kwargs):
        super().__init__(**kwargs)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self._map_name = map_name
        self._map_config = map_config
        self._supply_depot_positions = None
        self._command_center_positions = None
        self._barrack_positions = None
        self._used_supply_depot_positions = None
        self._used_command_center_positions = None
        self._used_barrack_positions = None
        self._attempted_supply_depot_positions = None
        self._attempted_command_center_positions = None
        self._attempted_barrack_positions = None
        self._train = train
        self._exploit = not train
        self._tracker: BaseEmissionsTracker = None
        self._tracker_update_freq_seconds = tracker_update_freq_seconds
        self._tracker_last_update = time.time()
        self._prev_action = None
        self._prev_action_args = None
        self._prev_action_is_valid = None
        self._prev_score = 0.
        self._current_score = 0.
        self._current_reward = 0.
        self._current_adjusted_reward = 0.
        self._reward_method = reward_method
        self._buffer = buffer
        self._current_state_tensor = None
        self._current_state_tuple = None
        self._prev_state_tensor = None
        self._prev_state_tuple = None

        self._action_to_idx = None
        self._idx_to_action = None
        self._num_actions = None
        self._best_mean_rewards = None

        if checkpoint_path is not None:
            self._checkpoint_path = Path(checkpoint_path)
            self._checkpoint_path.mkdir(exist_ok=True, parents=True)
            self._agent_path = self._checkpoint_path / self._AGENT_FILE
            self._buffer_path = self._checkpoint_path / self._BUFFER_FILE
            self._stats_path = self._checkpoint_path / self._STATS_FILE
        else:
            self._checkpoint_path = None
            self._main_network_path = None
            self._target_network_path = None
            self._agent_path = None
            self._buffer_path = None
            self._stats_path = None

        self._status_flags = dict(
            train_started=False,
            exploit_started=False,
        )

        self.initialize()

    @property
    def _collect_stats(self) -> bool:
        return True

    def setup_actions(self):
        self._idx_to_action = { idx: action for idx, action in enumerate(self.agent_actions) }
        self._action_to_idx = { action: idx for idx, action in enumerate(self.agent_actions) }
        self._num_actions = len(self.agent_actions)

    def initialize(self):
        self._current_episode_stats = EpisodeStats(map_name=self._map_name)
        self._episode_stats = {self._map_name: []}
        self._aggregated_episode_stats = {self._map_name: AggregatedEpisodeStats(map_name=self._map_name)}
        self._agent_stats = {self._map_name: AgentStats(map_name=self._map_name)}

    def set_tracker(self, tracker: BaseEmissionsTracker):
        self._tracker = tracker
        self._tracker_last_update = time.time()

    @property
    def current_agent_stats(self) -> AgentStats:
        return self._agent_stats[self._map_name]

    @property
    def current_aggregated_episode_stats(self) -> AggregatedEpisodeStats:
        return self._aggregated_episode_stats[self._map_name]

    @property
    def checkpoint_path(self) -> Optional[Path]:
        return self._checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, checkpoint_path: Union[str|Path]):
        checkpoint_path = Path(checkpoint_path)
        self._checkpoint_path = checkpoint_path
        self._update_checkpoint_paths()

    def _update_checkpoint_paths(self):
        if self.checkpoint_path is None:
            self._agent_path = None
            self._buffer_path = None
            self._stats_path = None
        else:
            self._agent_path = self.checkpoint_path / self._AGENT_FILE
            self._buffer_path = self.checkpoint_path / self._BUFFER_FILE
            self._stats_path = self.checkpoint_path / self._STATS_FILE

    def save(self, checkpoint_path: Union[str|Path] = None):
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
        elif self.checkpoint_path is None:
            raise RuntimeError(f"The agent's checkpoint path is None, and no checkpoint path was provided to 'save'. Please provide one of the two.")

        agent_attrs = self._get_agent_attrs()

        with open(self._agent_path, "wb") as f:
            pickle.dump(agent_attrs, f)
            self.logger.info(f"Saved agent attributes to {self._agent_path}")

        self.save_stats(self.checkpoint_path)

        if self._buffer is not None:
            with open(self._buffer_path, "wb") as f:
                pickle.dump(self._buffer, f)
                self.logger.info(f"Saved memory replay buffer to {self._buffer_path}")

    @classmethod
    def load(cls, checkpoint_path: Union[str|Path], map_name: str, map_config: Dict, buffer: ExperienceReplayBuffer = None, **kwargs) -> Self:
        checkpoint_path = Path(checkpoint_path)
        agent_attrs_file = checkpoint_path / cls._AGENT_FILE
        with open(agent_attrs_file, mode="rb") as f:
            agent_attrs = pickle.load(f)
        if "main_network_path" in agent_attrs:
            agent_attrs["main_network_path"] = checkpoint_path / cls._MAIN_NETWORK_FILE
        if "target_network_path" in agent_attrs:
            agent_attrs["target_network_path"] = checkpoint_path / cls._TARGET_NETWORK_FILE

        if buffer is not None:
            agent_attrs["buffer"] = buffer

        init_attrs = cls._extract_init_arguments(checkpoint_path=checkpoint_path, agent_attrs=agent_attrs, map_name=map_name, map_config=map_config)
        agent = cls(**init_attrs, **kwargs)
        agent._load_agent_attrs(agent_attrs)

        if agent._current_episode_stats is None or agent._current_episode_stats.map_name != map_name:
            agent._current_episode_stats = EpisodeStats(map_name=map_name)
        if map_name not in agent._agent_stats:
            agent._agent_stats[map_name] = AgentStats(map_name=map_name)
        if map_name not in agent._aggregated_episode_stats:
            agent._aggregated_episode_stats[map_name] = AggregatedEpisodeStats(map_name=map_name)
        if map_name not in agent._episode_stats:
            agent._episode_stats[map_name] = []

        return agent

    def save_stats(self, checkpoint_path: Union[str|Path] = None):
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
        elif self.checkpoint_path is None:
            raise RuntimeError(f"The agent's checkpoint path is None, and no checkpoint path was provided to 'save'. Please provide one of the two.")

        def _add_dummy_action(stats, count_cols):
            for count_col in count_cols:
                stats[count_col] = stats[count_col].apply(lambda d: dict(dummy=0) if len(d) == 0 else d)
            return stats

        try:
            all_episode_stats = [v for v in self._episode_stats.values()]
            all_episode_stats = reduce(lambda v1, v2: v1 + v2, all_episode_stats)
            episode_stats_pd = pd.DataFrame(data=all_episode_stats)
            episode_stats_pd = _add_dummy_action(episode_stats_pd, ["invalid_action_counts", "valid_action_counts"])
            episode_stats_pd.to_parquet(self._checkpoint_path / "episode_stats.parquet")
            self.logger.info(f"Saved episode stats")
        except Exception as error:
            self.logger.error(f"Error saving episode stats")
            self.logger.exception(error)

        try:
            all_agent_stats = [v for v in self._agent_stats.values()]
            agent_stats_pd = pd.DataFrame(data=all_agent_stats)
            agent_stats_pd = _add_dummy_action(
                agent_stats_pd,
                ["invalid_action_counts", "valid_action_counts", "invalid_action_counts_per_stage", "valid_action_counts_per_stage"])
            agent_stats_pd.to_parquet(self._checkpoint_path / "agent_stats.parquet")
            self.logger.info(f"Saved agent stats")
        except Exception as error:
            self.logger.error(f"Error saving agent stats")
            self.logger.exception(error)

        try:
            all_aggregated_stats = [v for v in self._aggregated_episode_stats.values()]
            aggregated_stats_pd = pd.DataFrame(data=all_aggregated_stats)
            aggregated_stats_pd = _add_dummy_action(
                aggregated_stats_pd,
                ["invalid_action_counts", "valid_action_counts", "invalid_action_counts_per_stage", "valid_action_counts_per_stage"])
            aggregated_stats_pd.to_parquet(self._checkpoint_path / "aggregated_stats.parquet")
            self.logger.info(f"Saved aggregated stats")
        except Exception as error:
            self.logger.error(f"Error saving aggregated stats")
            self.logger.exception(error)

    @classmethod
    def _extract_init_arguments(cls, checkpoint_path: Path, agent_attrs: Dict[str, Any], map_name: str, map_config: Dict) -> Dict[str, Any]:
        return dict(
            checkpoint_path=checkpoint_path,
            map_name=map_name,
            map_config=map_config,
            train=agent_attrs["train"],
            log_name=agent_attrs["log_name"],
            buffer=agent_attrs["buffer"],
        )

    def _load_agent_attrs(self, agent_attrs: Dict):
        self._train = agent_attrs["train"]
        self._exploit = agent_attrs.get("exploit", not self._train)
        self.checkpoint_path = agent_attrs["checkpoint_path"]
        self._agent_stats = agent_attrs["agent_stats"]
        self._episode_stats = agent_attrs["episode_stats"]
        self._aggregated_episode_stats = agent_attrs["aggregated_episode_stats"]
        # From SC2's Base agent
        self.reward = agent_attrs["reward"]
        self.episodes = agent_attrs["episodes"]
        self.steps = agent_attrs["steps"]
        self.obs_spec = agent_attrs["obs_spec"]
        self.action_spec = agent_attrs["action_spec"]
        self._reward_method = agent_attrs.get("reward_method", RewardMethod.REWARD)
        # From WithLogger
        self._log_name = agent_attrs["log_name"]
        if self.logger.name != self._log_name:
            self._logger = logging.getLogger(self._log_name)

    def _get_agent_attrs(self):
        return dict(
            train=self._train,
            exploit=self._exploit,
            checkpoint_path=self.checkpoint_path,
            buffer=self._buffer,
            agent_path=self._agent_path,
            stats_path=self._stats_path,
            agent_stats=self._agent_stats,
            episode_stats=self._episode_stats,
            aggregated_episode_stats=self._aggregated_episode_stats,
            # From SC2's Base agent
            reward=self.reward,
            episodes=self.episodes,
            steps=self.steps,
            obs_spec=self.obs_spec,
            action_spec=self.action_spec,
            reward_method=self._reward_method,
            # From logger
            log_name=self._log_name,
        )

    def train(self):
        """Set the agent in training mode."""
        self._train = True
        self._exploit = False

    def exploit(self):
        """Set the agent in training mode."""
        self._train = False
        self._exploit = True

    def _current_agent_stage(self):
        if self._exploit:
            return AgentStage.EXPLOIT
        if self.is_training:
            return AgentStage.TRAINING
        return AgentStage.UNKNOWN

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._supply_depot_positions = None
        self._barrack_positions = None
        self._command_center_positions = None
        self._used_supply_depot_positions = None
        self._used_command_center_positions = None
        self._used_barrack_positions = None
        self._attempted_supply_depot_positions = None
        self._attempted_command_center_positions = None
        self._attempted_barrack_positions = None
        self._prev_action = None
        self._prev_action_args = None
        self._prev_action_is_valid = None
        self._prev_score = 0.
        self._current_score = 0.
        self._current_reward = 0.
        self._current_adjusted_reward = 0.
        self._prev_state_tensor = None
        self._prev_state_tuple = None
        self._current_state_tensor = None
        self._current_state_tuple = None

        current_stage = self._current_agent_stage().name
        self._current_episode_stats = EpisodeStats(map_name=self._map_name, is_burnin=False, is_training=self.is_training, is_exploit=self._exploit, episode=self.current_agent_stats.episode_count, initial_stage=current_stage)
        self.current_agent_stats.episode_count += 1
        self.current_agent_stats.episode_count_per_stage[current_stage] += 1

    @property
    def is_training(self):
        return self._train

    def _setup_positions(self, obs: TimeStep):
        match self._map_name:
            case "Simple64":
                command_center = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)[0]
                position = "top_left" if command_center.y < 50 else "bottom_right"
                self.logger.info(f"Map {self._map_name} - Started at '{position}' position")
                self._supply_depot_positions = self._map_config["positions"][position].get(units.Terran.SupplyDepot, []).copy()
                self._command_center_positions = self._map_config["positions"][position].get(units.Terran.CommandCenter, []).copy()
                self._barrack_positions = self._map_config["positions"][position].get(units.Terran.Barracks, []).copy()
            case "CollectMineralsAndGas":
                self._supply_depot_positions = self._map_config["positions"].get(units.Terran.SupplyDepot, []).copy()
                self._command_center_positions = self._map_config["positions"].get(units.Terran.CommandCenter, []).copy()
                self._barrack_positions = self._map_config["positions"].get(units.Terran.Barracks, []).copy()
            case _ if not self._map_config["multiple_positions"]:
                self._supply_depot_positions = self._map_config["positions"].get(units.Terran.SupplyDepot, []).copy()
                self._command_center_positions = self._map_config["positions"].get(units.Terran.CommandCenter, []).copy()
                self._barrack_positions = self._map_config["positions"].get(units.Terran.Barracks, []).copy()
            case _:
                raise RuntimeError(f"Map {self._map_name} has multiple positions, but no logic to determine which positions to take")

        self._supply_depot_positions = [Position(t[0], t[1]) for t in self._supply_depot_positions]
        self._command_center_positions = [Position(t[0], t[1]) for t in self._command_center_positions]
        self._barrack_positions = [Position(t[0], t[1]) for t in self._barrack_positions]
        self._attempted_barrack_positions = []
        self._attempted_supply_depot_positions = []
        self._attempted_command_center_positions = []
        self.update_supply_depot_positions(obs)
        self.update_command_center_positions(obs)
        self.update_barracks_positions(obs)

    @property
    @abstractmethod
    def agent_actions(self) -> List[AllActions]:
        pass

    @abstractmethod
    def select_action(self, obs: TimeStep) -> Tuple[AllActions, Dict[str, Any]]:
        pass
        # return actions.FUNCTIONS.no_op()

    def _get_action_args(self, obs: TimeStep, action: AllActions) -> Dict[str, any]:
        is_valid = True
        try:
            match action:
                case AllActions.NO_OP:
                    action_args = None
                case AllActions.HARVEST_MINERALS:
                    minerals = [unit for unit in obs.observation.raw_units if Minerals.contains(unit.unit_type)]
                    command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
                    idle_workers = self.get_idle_workers(obs)
                    assert (len(minerals) > 0), "There are no minerals to harvest"
                    assert (len(command_centers) > 0), "There are no command centers"
                    assert (len(idle_workers) > 0), "There are no idle workers"

                    worker = random.choice(idle_workers)
                    command_center, _ = self.get_closest(command_centers, Position(worker.x, worker.y))
                    mineral, _ = self.get_closest(minerals, Position(command_center.x, command_center.y))
                    action_args = dict(source_unit_tag=worker.tag, target_unit_tag=mineral.tag)
                case AllActions.RECRUIT_SCV:
                    command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
                    command_centers = [cc for cc in command_centers if cc.order_length < Constants.COMMAND_CENTER_QUEUE_LENGTH]
                    assert len(command_centers) > 0, "There are no available command centers"
                    command_centers = sorted(command_centers, key=lambda cc: cc.order_length)
                    action_args = dict(source_unit_tag=random.choice(command_centers).tag)
                case AllActions.BUILD_SUPPLY_DEPOT:
                    position = self.get_next_supply_depot_position(obs)
                    workers = self.get_idle_workers(obs)
                    if len(workers) == 0:
                        workers = self.get_harvester_workers(obs)
                    assert position is not None, "The next supply depot position is None"
                    assert len(workers) > 0, "There are no workers to build the supply depot"
                    worker, _ = self.get_closest(workers, position)
                    action_args = dict(source_unit_tag=worker.tag, target_position=position)
                case AllActions.BUILD_COMMAND_CENTER:
                    position = self.get_next_command_center_position(obs)
                    workers = self.get_idle_workers(obs)
                    if len(workers) == 0:
                        workers = self.get_harvester_workers(obs)
                    assert position is not None, "The next command center position is None"
                    assert len(workers) > 0, "There are no workers to build the command center"
                    worker, _ = self.get_closest(workers, position)
                    action_args = dict(source_unit_tag=worker.tag, target_position=position)
                case AllActions.BUILD_BARRACKS:
                    position = self.get_next_barracks_position(obs)
                    workers = self.get_idle_workers(obs)
                    if len(workers) == 0:
                        workers = self.get_harvester_workers(obs)
                    assert position is not None, "The next barracks position is None"
                    assert len(workers) > 0, "There are no workers to build the barracks"
                    worker, _ = self.get_closest(workers, position)
                    action_args = dict(source_unit_tag=worker.tag, target_position=position)
                case AllActions.RECRUIT_MARINE:
                    barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks)
                    barracks = [cc for cc in barracks if cc.order_length < Constants.BARRACKS_QUEUE_LENGTH]
                    assert len(barracks) > 0, "There are no barracks available"
                    action_args = dict(source_unit_tag=random.choice(barracks).tag)
                case AllActions.ATTACK_WITH_SINGLE_UNIT:
                    idle_marines = self.get_idle_marines(obs)
                    assert len(idle_marines) > 0, "There are no idle marines"
                    enemies = self.get_enemy_units(obs, unit_types=Constants.COMMAND_CENTER_UNIT_TYPES)
                    if len(enemies) == 0:
                        enemies = self.get_enemy_units(obs, unit_types=Constants.BUILDING_UNIT_TYPES)
                    if len(enemies) == 0:
                        enemies = self.get_enemy_units(obs)
                    assert len(enemies) > 0, "There are no enemies"
                    enemy_positions = [Position(e.x, e.y) for e in enemies]
                    mean_enemy_position = np.mean(enemy_positions, axis=0)
                    target_position = Position(int(mean_enemy_position[0]), int(mean_enemy_position[1]))
                    closest_enemy, _ = self.get_closest(enemies, target_position)
                    target_position = Position(closest_enemy.x, closest_enemy.y)
                    marine_tag = random.choice(idle_marines).tag
                    action_args = dict(source_unit_tag=marine_tag, target_position=target_position)
                case _:
                    raise RuntimeError(f"Missing logic to select action args for action {action.name}")

            return action_args, True
        except AssertionError as error:
            self.logger.debug(error)
            return None, False

    def get_next_command_center_position(self, obs: TimeStep) -> Position:
        next_pos = None
        command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
        command_centers_positions = [Position(cc.x, cc.y) for cc in command_centers]

        enemy_command_centers = self.get_enemy_units(obs, unit_types=Constants.COMMAND_CENTER_UNIT_TYPES)
        enemy_command_centers_positions = [Position(cc.x, cc.y) for cc in enemy_command_centers]
        all_cc_positions = command_centers_positions + enemy_command_centers_positions
        for idx, candidate_position in enumerate(self._command_center_positions):
            if (candidate_position not in self._used_command_center_positions) and (candidate_position not in all_cc_positions):
                next_pos = candidate_position
                break

        return next_pos

    def update_command_center_positions(self, obs: TimeStep) -> Position:
        command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
        command_center_positions = [Position(cc.x, cc.y) for cc in command_centers]
        enemy_command_centers = self.get_enemy_units(obs, unit_types=Constants.COMMAND_CENTER_UNIT_TYPES)
        enemy_command_center_positions = [Position(cc.x, cc.y) for cc in enemy_command_centers]
        self._used_command_center_positions = command_center_positions + enemy_command_center_positions
        # self._command_center_positions = [pos for pos in self._command_center_positions if pos not in command_center_positions]

    def use_command_center_position(self, obs: TimeStep, position: Position) -> Position:
        if position not in self._command_center_positions:
            return False

        idx = self._command_center_positions.index(position)
        self._command_center_positions = self._command_center_positions[idx + 1:] + self._command_center_positions[:idx+1]

        return True

    def get_next_supply_depot_position(self, obs: TimeStep) -> Position:
        next_pos = None
        for idx, candidate_position in enumerate(self._supply_depot_positions):
            if candidate_position not in self._used_supply_depot_positions:
                next_pos = candidate_position
                break

        return next_pos

    def update_supply_depot_positions(self, obs: TimeStep) -> Position:
        supply_depots = self.get_self_units(obs, unit_types=units.Terran.SupplyDepot)
        supply_depots_positions = [Position(sd.x, sd.y) for sd in supply_depots]
        self._used_supply_depot_positions = supply_depots_positions

    def use_supply_depot_position(self, obs: TimeStep, position: Position) -> Position:
        if position not in self._supply_depot_positions:
            return False

        idx = self._supply_depot_positions.index(position)
        self._supply_depot_positions = self._supply_depot_positions[idx + 1:] + self._supply_depot_positions[:idx+1]

        return True

    def get_next_barracks_position(self, obs: TimeStep) -> Position:
        next_pos = None
        for idx, candidate_position in enumerate(self._barrack_positions):
            if candidate_position not in self._used_barrack_positions:
                next_pos = candidate_position
                break

        return next_pos

    def use_barracks_position(self, obs: TimeStep, position: Position) -> Position:
        if position not in self._barrack_positions:
            return False

        idx = self._barrack_positions.index(position)
        self._barrack_positions = self._barrack_positions[idx + 1:] + self._barrack_positions[:idx+1]

        return True

    def update_barracks_positions(self, obs: TimeStep) -> Position:
        barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks)
        barrack_positions = [Position(b.x, b.y) for b in barracks]
        self._used_barrack_positions = barrack_positions

    def get_reward_and_score(self, obs: TimeStep) -> Tuple[float, float, float]:
        reward = obs.reward

        get_score = getattr(self, self._map_config["get_score_method"])
        score = get_score(obs)
        adjusted_reward = reward + Constants.STEP_REWARD

        if not (obs.last() or obs.first()):
            if self._prev_action_is_valid == False:
                adjusted_reward = Constants.INVALID_ACTION_REWARD
            elif self._prev_action == AllActions.NO_OP:
                adjusted_reward = Constants.NO_OP_REWARD

        self._current_score = score
        self._current_reward = reward
        self._current_adjusted_reward = adjusted_reward

        return reward, adjusted_reward, score

    def get_army_health_difference(self, obs: TimeStep) -> float:
        enemy_army = self.get_enemy_units(obs, unit_types=Constants.ARMY_UNIT_TYPES)
        enemy_total_army_health = sum(map(lambda b: b.health, enemy_army))
        marines = self.get_self_units(obs, unit_types=units.Terran.Marine)
        total_army_health = sum(map(lambda b: b.health, marines))

        return total_army_health - enemy_total_army_health

    def get_mineral_collection_rate_difference(self, obs: TimeStep) -> float:
        return obs.observation.score_cumulative.collection_rate_minerals / 60

    def get_num_marines_difference(self, obs: TimeStep) -> float:
        if not obs.first():
            prev_num_marines = self._prev_state_tuple.num_marines
        else:
            prev_num_marines = 0

        marines = self.get_self_units(obs, unit_types=units.Terran.Marine)
        num_marines = len(marines)

        return num_marines - prev_num_marines + Constants.STEP_REWARD

    def get_reward_as_score(self, obs: TimeStep) -> float:
        return obs.reward

    def _convert_obs_to_state(self, obs: TimeStep) -> torch.Tensor:
        actions_state = self._get_actions_state(obs)
        building_state = self._get_buildings_state(obs)
        worker_state = self._get_workers_state(obs)
        army_state = self._get_army_state(obs)
        resources_state = self._get_resources_state(obs)
        scores_state = self._get_scores_state(obs)
        neutral_units_state = self._get_neutral_units_state(obs)
        enemy_state = self._get_enemy_state(obs)
        # Enemy

        state_tuple = State(
            **actions_state,
			**building_state,
			**worker_state,
			**army_state,
			**resources_state,
            **scores_state,
            **neutral_units_state,
            **enemy_state
        )
        return torch.Tensor(state_tuple).to(device=self.device), state_tuple

    def _actions_to_network(self, actions: List[AllActions], as_tensor: bool = True) -> List[np.int8]:
        """Converts a list of AllAction elements to a one-hot encoded version that the network can use.

        Args:
            actions (List[AllActions]): List of actions

        Returns:
            List[bool]: One-hot encoded version of the actions provided.
        """
        ohe_actions = np.zeros(self._num_actions, dtype=np.int8)

        for action in actions:
            action_idx = self._action_to_idx[action]
            ohe_actions[action_idx] = 1

        if not as_tensor:
            return ohe_actions
        return torch.Tensor(ohe_actions).to(device=self.device)

    def pre_step(self, obs: TimeStep):
        self._current_state_tensor, self._current_state_tuple = self._convert_obs_to_state(obs)

        if not self._exploit:
            reward, adjusted_reward, score = self.get_reward_and_score(obs)
            self._current_episode_stats.reward += reward
            self._current_episode_stats.adjusted_reward += adjusted_reward
            self._current_episode_stats.score += score
            self._current_episode_stats.steps += 1
            self.current_agent_stats.step_count += 1

        if obs.first():
            self.setup_actions()
        elif not self._exploit:
            done = obs.last()
            if self._buffer is not None:
                self._buffer.append(self._prev_state_tensor, self._prev_action, self._prev_action_args, self._current_reward, self._current_adjusted_reward, self._current_score, done, self._current_state_tensor)

    def post_step(self, obs: TimeStep, action: AllActions, action_args: Dict[str, Any], original_action: AllActions, original_action_args: Dict[str, Any], is_valid_action: bool):
        self._prev_score = obs.observation.score_cumulative.score
        self._prev_state_tensor = self._current_state_tensor
        self._prev_state_tuple = self._current_state_tuple
        self._prev_action = self._action_to_idx[original_action]
        self._prev_action_args = original_action_args

        if not self._exploit:
            if obs.last():
                emissions = self._tracker.flush() if self._tracker is not None else 0.
                self.logger.debug(f"End of episode - Got extra {emissions} since last update")
                self._current_episode_stats.emissions += emissions
                self._current_episode_stats.is_training = self.is_training
                self._current_episode_stats.is_exploit = self._exploit
                self._current_episode_stats.final_stage = self._current_agent_stage().name
                self._tracker_last_update = time.time()
                episode_stage = self._current_episode_stats.initial_stage
                mean_rewards = self.current_aggregated_episode_stats.mean_rewards(stage=episode_stage, reward_method=self._reward_method)
                max_mean_rewards = self._best_mean_rewards
                max_mean_rewards_str = f"{max_mean_rewards:.2f}" if max_mean_rewards is not None else "None"
                new_max_mean_rewards = (max_mean_rewards is None) or (mean_rewards >= max_mean_rewards)

                if (self.is_training) and (self.checkpoint_path is not None) and new_max_mean_rewards:
                    self.logger.info(f"New max reward during training ({max_mean_rewards_str} -> {mean_rewards:.2f}). Saving best agent...")
                    checkpoint_path = self.checkpoint_path
                    save_path = self.checkpoint_path / "best_agent"
                    save_path.mkdir(exist_ok=True, parents=True)
                    self.save(checkpoint_path=save_path)
                    self.checkpoint_path = checkpoint_path
                    self._best_mean_rewards = mean_rewards

                self.current_agent_stats.process_episode(self._current_episode_stats)
                self.current_aggregated_episode_stats.process_episode(self._current_episode_stats)
                self._episode_stats[self._map_name].append(self._current_episode_stats)
                log_msg_parts = ["\n=================", "================="] + self._get_end_of_episode_info_components() + ["=================", "================="]
                log_msg = "\n".join(log_msg_parts)
                self.logger.info(log_msg)
            else:
                now = time.time()
                if (self._tracker is not None) and (now - self._tracker_last_update > self._tracker_update_freq_seconds):
                    emissions = self._tracker.flush() or 0.
                    self.logger.debug(f"Tracker flush - Got extra {emissions} since last update")
                    self._current_episode_stats.emissions += emissions
                    self._tracker_last_update = now

    def _get_end_of_episode_info_components(self) -> List[str]:
        num_invalid = sum(self._current_episode_stats.invalid_action_counts.values())
        num_valid = sum(self._current_episode_stats.valid_action_counts.values())
        pct_invalid = num_invalid / (num_invalid + num_valid)
        episode_stage = self._current_episode_stats.initial_stage
        mean_rewards = self.current_aggregated_episode_stats.mean_rewards(stage=episode_stage, reward_method=RewardMethod.REWARD)
        mean_rewards_10 = self.current_aggregated_episode_stats.mean_rewards(stage=episode_stage, last_n=10, reward_method=RewardMethod.REWARD)
        mean_adjusted_rewards = self.current_aggregated_episode_stats.mean_rewards(stage=episode_stage, reward_method=RewardMethod.ADJUSTED_REWARD)
        mean_adjusted_rewards_10 = self.current_aggregated_episode_stats.mean_rewards(stage=episode_stage, last_n=10, reward_method=RewardMethod.ADJUSTED_REWARD)
        mean_scores = self.current_aggregated_episode_stats.mean_rewards(stage=episode_stage, reward_method=RewardMethod.SCORE)
        mean_scores_10 = self.current_aggregated_episode_stats.mean_rewards(stage=episode_stage, last_n=10, reward_method=RewardMethod.SCORE)
        episode_count = self.current_agent_stats.episode_count_per_stage[episode_stage]
        return [
            f"Episode {self._map_name} // Stage: {episode_stage} // Final stage: {self._current_agent_stage().name}",
            f"Reward method: {self._reward_method.name}",
            f"Reward: {self._current_episode_stats.reward} // Score: {self._current_episode_stats.score}",
            f"Episode {episode_count}",
            f"Mean Rewards for stage ({episode_count} ep) {mean_rewards:.2f} / (10ep) {mean_rewards_10:.2f}",
            f"Mean Adjusted Rewards for stage ({episode_count} ep) {mean_adjusted_rewards:.2f} / (10ep) {mean_adjusted_rewards_10:.2f}",
            f"Mean Scores ({episode_count} ep) {mean_scores:.2f} / (10ep) {mean_scores_10:.2f}",
            f"Episode steps: {self._current_episode_stats.steps} / Total steps: {self.current_agent_stats.step_count_per_stage[episode_stage]}",
            f"Invalid actions: {num_invalid}/{num_valid + num_invalid} ({100 * pct_invalid:.2f}%)",
            f"Max reward {self.current_aggregated_episode_stats.max_reward_per_stage[episode_stage]:.2f} (absolute max: {self.current_aggregated_episode_stats.max_reward})",
            f"Max adjusted reward {self.current_aggregated_episode_stats.max_adjusted_reward_per_stage[episode_stage]:.2f} (absolute max: {self.current_aggregated_episode_stats.max_adjusted_reward})",
            f"Max score {self.current_aggregated_episode_stats.max_score_per_stage[episode_stage]:.2f} (absolute max: {self.current_aggregated_episode_stats.max_score})",
            f"Episode emissions: {self._current_episode_stats.emissions} / Total in stage: {self.current_agent_stats.total_emissions_per_stage[episode_stage]} / Total: {self.current_agent_stats.total_emissions}",
        ]

    def step(self, obs: TimeStep, only_super_step: bool = False) -> AllActions:
        if only_super_step:
            super().step(obs)
            return
        if obs.first():
            self._setup_positions(obs)
        self.pre_step(obs)

        super().step(obs)

        self.update_supply_depot_positions(obs)
        self.update_command_center_positions(obs)
        self.update_barracks_positions(obs)

        action, action_args, is_valid_action = self.select_action(obs)
        original_action = action
        original_action_args = action_args

        if not is_valid_action:
            self.logger.debug(f"Action {action.name} is not valid anymore, returning NO_OP")
            self._current_episode_stats.add_invalid_action(action)
            action = AllActions.NO_OP
            action_args = None
        elif action == AllActions.BUILD_BARRACKS:
            self.use_barracks_position(obs, action_args["target_position"])
        elif action == AllActions.BUILD_SUPPLY_DEPOT:
            self.use_supply_depot_position(obs, action_args["target_position"])
        elif action == AllActions.BUILD_COMMAND_CENTER:
            self.use_command_center_position(obs, action_args["target_position"])

        if is_valid_action:
            self._current_episode_stats.add_valid_action(action)
            self.logger.debug(f"[Step {self.steps}] Performing action {action.name} with args: {action_args}")
        game_action = self._action_to_game[action]

        self.post_step(obs, action, action_args, original_action, original_action_args, is_valid_action)

        if action_args is not None:
            return game_action(**action_args)

        return game_action()

    def available_actions(self, obs: TimeStep) -> List[AllActions]:
        available_actions = [a for a in self.agent_actions if self.can_take(obs, a)]
        if len(available_actions) > 1 and AllActions.NO_OP in available_actions:
            available_actions = [a for a in available_actions if a != AllActions.NO_OP]

        return available_actions

    def get_idle_marines(self, obs: TimeStep) -> List[features.FeatureUnit]:
        """Gets all idle workers.

        Args:
            obs (TimeStep): Observation from the environment

        Returns:
            List[features.FeatureUnit]: List of idle workers
        """
        self_marines = self.get_self_units(obs, units.Terran.Marine)
        idle_marines = filter(self.is_idle, self_marines)

        return list(idle_marines)

    def has_marines(self, obs: TimeStep) -> bool:
        return len(self.get_self_units(obs, units.Terran.Marine)) > 0

    def has_idle_marines(self, obs: TimeStep) -> bool:
        return len(self.get_idle_marines(obs)) > 0

    def has_idle_workers(self, obs: TimeStep) -> bool:
        return len(self.get_idle_workers(obs)) > 0

    def has_harvester_workers(self, obs: TimeStep) -> bool:
        return len(self.get_harvester_workers(obs)) > 0

    def has_workers(self, obs: TimeStep) -> bool:
        return len(self.get_self_units(obs, units.Terran.SCV)) > 0

    def can_take(self, obs: TimeStep, action: AllActions, **action_args) -> bool:
        if action == AllActions.NO_OP:
            return True
        if action not in self.agent_actions:
            return False
        elif action not in self._map_config["available_actions"]:
            return False
        elif action not in self._action_to_game:
            self.logger.warning(f"Tried to validate action {action.name} that is not yet implemented in the action to game mapper: {self._action_to_game.keys()}")
            return False

        def _has_target_unit_tag(args):
            return "target_unit_tag" in args and isinstance(args["target_unit_tag"], np.int64)

        def _has_source_unit_tag(args):
            return "source_unit_tag" in args and isinstance(args["source_unit_tag"], np.int64)

        def _has_source_unit_tags(args, of_length: int = None):
            if "source_unit_tags" not in args:
                return False
            if any(map(lambda t: not isinstance(t, np.int64), args["source_unit_tags"])):
                return False
            if of_length is not None and len(args["source_unit_tags"]) != of_length:
                return False

            return True

        match action, action_args:
            case AllActions.NO_OP, _:
                return True
            case AllActions.HARVEST_MINERALS, args if _has_target_unit_tag(args):
                target_unit_tag = args["target_unit_tag"]
                command_centers = [unit.tag for unit in self.get_self_units(obs, unit_types=units.Terran.CommandCenter)]
                if not any(command_centers):
                    return False

                minerals = [unit.tag for unit in obs.observation.raw_units if unit.tag == target_unit_tag and Minerals.contains(unit.unit_type)]
                if len(minerals) == 0:
                    return False

                if self.has_idle_workers(obs):
                    return True
                return False
            case AllActions.HARVEST_MINERALS, _:
                command_centers = [unit.tag for unit in self.get_self_units(obs, unit_types=units.Terran.CommandCenter)]

                if not any(command_centers):
                    return False

                minerals = [unit.tag for unit in obs.observation.raw_units if Minerals.contains(unit.unit_type)]
                if not any(minerals):
                    return False

                if self.has_idle_workers(obs):
                    return True
                return False
            case AllActions.RECRUIT_SCV, args if _has_source_unit_tag(args):
                command_center_tag = args["source_unit_tag"]
                command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter, unit_tags=command_center_tag)
                command_centers = [cc for cc in command_centers if cc.order_length < Constants.COMMAND_CENTER_QUEUE_LENGTH]
                if len(command_centers) == 0:
                    return False
                if command_centers[0].order_length >= Constants.COMMAND_CENTER_QUEUE_LENGTH:
                    return False
                if not SC2Costs.SCV.can_pay(obs.observation.player):
                    return False
                return True
            case AllActions.RECRUIT_SCV, _:
                command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
                if len(command_centers) == 0:
                    return False

                for command_center in command_centers:
                    if self.can_take(obs, action, source_unit_tag=command_center.tag):
                        return True
            case AllActions.BUILD_SUPPLY_DEPOT, _:
                target_position = self.get_next_supply_depot_position(obs)
                if target_position is None:
                    return False
                if not self.has_idle_workers(obs):
                    if not self.has_harvester_workers(obs):
                        return False
                if not SC2Costs.SUPPLY_DEPOT.can_pay(obs.observation.player):
                    return False

                return True
            case AllActions.BUILD_COMMAND_CENTER, _:
                target_position = self.get_next_command_center_position(obs)
                if target_position is None:
                    return False
                command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
                command_centers_positions = [Position(cc.x, cc.y) for cc in command_centers]
                if target_position in command_centers_positions:
                    return False

                enemy_command_centers = self.get_enemy_units(obs, unit_types=Constants.COMMAND_CENTER_UNIT_TYPES)
                enemy_command_centers_positions = [Position(cc.x, cc.y) for cc in enemy_command_centers]
                if target_position in enemy_command_centers_positions:
                    return False

                if not self.has_idle_workers(obs):
                    if not self.has_harvester_workers(obs):
                        return False

                if not SC2Costs.COMMAND_CENTER.can_pay(obs.observation.player):
                    return False

                return True
            case AllActions.BUILD_BARRACKS, _:
                target_position = self.get_next_barracks_position(obs)
                if target_position is None:
                    return False
                if not self.has_idle_workers(obs):
                    if not self.has_harvester_workers(obs):
                        return False
                supply_depots = self.get_self_units(obs, unit_types=units.Terran.SupplyDepot)
                if len(supply_depots) == 0:
                    return False

                if not SC2Costs.BARRACKS.can_pay(obs.observation.player):
                    return False

                return True
            case AllActions.RECRUIT_MARINE, args if _has_source_unit_tag(args):
                barracks_tag = args["source_unit_tag"]
                barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks, unit_tags=barracks_tag)
                barracks = [b for b in barracks if b.order_length < Constants.BARRACKS_QUEUE_LENGTH]
                if len(barracks) == 0:
                    return False

                if not SC2Costs.MARINE.can_pay(obs.observation.player):
                    return False
                return True

            case AllActions.RECRUIT_MARINE, _:
                barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks)
                barracks = [b for b in barracks if b.order_length < Constants.BARRACKS_QUEUE_LENGTH]
                if len(barracks) == 0:
                    return False

                if not SC2Costs.MARINE.can_pay(obs.observation.player):
                    return False

                return True
            case AllActions.ATTACK_WITH_SINGLE_UNIT, _:
                idle_marines = self.get_idle_marines(obs)
                if len(idle_marines) == 0:
                    return False
                enemies = self.get_enemy_units(obs)
                if len(enemies) == 0:
                    return False

                return True
            case _:
                self.logger.warning(f"Action {action.name} ({action}) is not implemented yet")
                return False

    def get_self_units(self, obs: TimeStep, unit_types: int | List[int] = None, unit_tags: int | List[int] = None, completed_only: bool = False) -> List[features.FeatureUnit]:
        """Get a list of the player's own units.

        Args:
            obs (TimeStep): Observation from the environment
            unit_type (int | List[int], optional): Type of unit(s) to get. If provided, only units of this type(s) will be
                                       returned, otherwise all units are returned.

        Returns:
            List[features.FeatureUnit]: List of player units, filtered by unit type and/or tag if provided
        """
        units = filter(lambda u: u.alliance == PlayerRelative.SELF, obs.observation.raw_units)

        if unit_types is not None:
            unit_types = [unit_types] if isinstance(unit_types, int) else unit_types
            units = filter(lambda u: u.unit_type in unit_types, units)

        if unit_tags is not None:
            unit_tags = [unit_tags] if isinstance(unit_tags, (int, np.int64, np.integer, np.int32)) else unit_tags
            units = filter(lambda u: u.tag in unit_tags, units)

        if completed_only:
            units = filter(lambda u: u.build_progress == 100, units)

        return list(units)

    def get_neutral_units(self, obs: TimeStep, unit_types: int | List[int] = None, unit_tags: int | List[int] = None) -> List[features.FeatureUnit]:
        """Get a list of neutral units.

        Args:
            obs (TimeStep): Observation from the environment
            unit_type (int | List[int], optional): Type of unit(s) to get. If provided, only units of this type(s) will be
                                       returned, otherwise all units are returned.

        Returns:
            List[features.FeatureUnit]: List of neutral units, filtered by unit type and/or tag if provided
        """
        units = filter(lambda u: u.alliance == PlayerRelative.ENEMY, obs.observation.raw_units)

        if unit_types is not None:
            unit_types = [unit_types] if isinstance(unit_types, int) else unit_types
            units = filter(lambda u: u.unit_type in unit_types, units)

        if unit_tags is not None:
            unit_tags = [unit_tags] if isinstance(unit_tags, (int, np.int64, np.integer, np.int32)) else unit_tags
            units = filter(lambda u: u.tag in unit_tags, units)

        return list(units)

    def get_enemy_units(self, obs: TimeStep, unit_types: int | List[int] = None, unit_tags: int | List[int] = None) -> List[features.FeatureUnit]:
        """Get a list of the player's enemy units.

        Args:
            obs (TimeStep): Observation from the environment
            unit_type (int | List[int], optional): Type of unit(s) to get. If provided, only units of this type(s) will be
                                       returned, otherwise all units are returned.

        Returns:
            List[features.FeatureUnit]: List of enemy units, filtered by unit type and/or tag if provided
        """
        units = filter(lambda u: u.alliance == PlayerRelative.ENEMY, obs.observation.raw_units)

        if unit_types is not None:
            unit_types = [unit_types] if isinstance(unit_types, int) else unit_types
            units = filter(lambda u: u.unit_type in unit_types, units)

        if unit_tags is not None:
            unit_tags = [unit_tags] if isinstance(unit_tags, (int, np.int64, np.integer, np.int32)) else unit_tags
            units = filter(lambda u: u.tag in unit_tags, units)

        return list(units)

    def get_enemies_info(self, obs: TimeStep) -> List[Dict]:
        return [
                dict(tag=e.tag, type=units.get_unit_type(e.unit_type), position=Position(e.x, e.y)) for e in self.get_enemy_units(obs)
            ]

    def get_workers(self, obs: TimeStep, idle: bool = False, harvesting: bool = False) -> List[features.FeatureUnit]:
        """Gets all idle workers.

        Args:
            obs (TimeStep): Observation from the environment

        Returns:
            List[features.FeatureUnit]: List of idle workers
        """
        if idle and harvesting:
            self.logger.warning(f"Asking for workers that are idle AND harvesting will always result in an empty list")
            return []

        workers = self.get_self_units(obs, units.Terran.SCV)

        if idle:
            workers = filter(self.is_idle, workers)
        elif harvesting:
            workers = filter(lambda w: w.order_id_0 in Constants.HARVEST_ACTIONS, workers)

        return list(workers)

    def get_idle_workers(self, obs: TimeStep) -> List[features.FeatureUnit]:
        """Gets all idle workers.

        Args:
            obs (TimeStep): Observation from the environment

        Returns:
            List[features.FeatureUnit]: List of idle workers
        """
        return self.get_workers(obs, idle=True)

    def get_harvester_workers(self, obs: TimeStep) -> List[features.FeatureUnit]:
        """Get a list of all workers that are currently harvesting.

        Args:
            obs (TimeStep): Observation from the environment.

        Returns:
            List[features.FeatureUnit]: List of workers that are harvesting.
        """
        return self.get_workers(obs, harvesting=True)

    def get_free_supply(self, obs: TimeStep) -> int:
        return obs.observation.player.food_cap - obs.observation.player.food_used

    def is_idle(self, unit: features.FeatureUnit) -> bool:
        """Check whether a unit is idle (meaning it has no orders in the queue)"""
        return unit.order_length == 0

    def is_complete(self, unit: features.FeatureUnit) -> bool:
        """Check whether a unit is fully build"""
        return unit.build_progress == 100

    def get_distances(self, units: List[features.FeatureUnit], position: Position) -> List[float]:
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(position), axis=1)

    def get_closest(self, units: List[features.FeatureUnit], position: Position) -> Tuple[features.FeatureUnit, float]:
        distances = self.get_distances(units, position)
        min_distance = distances.min()
        min_distances = np.where(distances == min_distance)[0]
        # If there is only one minimum distance, that will be returned, otherwise we return one of the elements with the minimum distance

        closes_unit_idx = np.random.choice(min_distances)
        return units[closes_unit_idx], min_distance

    def select_closest_worker_to_resource(self, obs: TimeStep, workers: List[features.FeatureUnit], resources: List[features.FeatureUnit]) -> Tuple[features.FeatureUnit, features.FeatureUnit]:
        closest_worker = None
        shortest_distance = None
        closest_resource = None
        for worker in workers:
            worker_position = Position(worker.x, worker.y)
            closest_resource, total_distance = self.get_closest(resources, worker_position)

            if closest_worker is None:
                closest_worker = worker
                shortest_distance = total_distance
                target_resource = closest_resource
            elif total_distance <= shortest_distance:
                closest_worker = worker
                shortest_distance = total_distance
                target_resource = closest_resource

        return closest_worker, target_resource

    def select_closest_worker(self, obs: TimeStep, workers: List[features.FeatureUnit], command_centers: List[features.FeatureUnit], resources: List[features.FeatureUnit]) -> Tuple[features.FeatureUnit, features.FeatureUnit]:
        command_center_distances = {}
        command_center_closest_resource = {}
        for command_center in command_centers:
            command_center_position = Position(command_center.x, command_center.y)
            closest_resource, distance = self.get_closest(resources, command_center_position)
            command_center_distances[command_center.tag] = distance
            command_center_closest_resource[command_center.tag] = closest_resource

        closest_worker = None
        shortest_distance = None
        closest_resource = None
        for worker in workers:
            worker_position = Position(worker.x, worker.y)
            closest_command_center, distance_to_command_center = self.get_closest(command_centers, worker_position)
            distance_to_resource = command_center_distances[closest_command_center.tag]
            total_distance = distance_to_command_center + distance_to_resource
            if closest_worker is None:
                closest_worker = worker
                shortest_distance = total_distance
                target_resource = command_center_closest_resource[closest_command_center.tag]
            elif total_distance <= shortest_distance:
                closest_worker = worker
                shortest_distance = total_distance
                target_resource = command_center_closest_resource[closest_command_center.tag]

        return closest_worker, target_resource


    def _get_buildings_state(self, obs):
        def _num_complete(buildings):
            return len(list(filter(self.is_complete, buildings)))
        # info about command centers
        command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
        num_command_centers = len(command_centers)
        num_completed_command_centers = _num_complete(command_centers)
        command_centers_state = dict(
            num_command_centers=num_command_centers,
            num_completed_command_centers=num_completed_command_centers
        )

        for idx in range(3):
            if idx >= num_command_centers:
                order_length = -1
                assigned_harvesters = -1
            else:
                cc = command_centers[idx]
                order_length = cc.order_length
                assigned_harvesters = cc.assigned_harvesters

            command_centers_state[f"command_center_{idx}_order_length"] = order_length
            command_centers_state[f"command_center_{idx}_num_workers"] = assigned_harvesters

        # Buildings
        supply_depots = self.get_self_units(obs, unit_types=units.Terran.SupplyDepot)
        num_supply_depots = len(supply_depots)
        barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks)
        num_barracks = len(barracks)

        buildings_state = dict(
			# Supply Depots
			num_supply_depots=num_supply_depots,
			num_completed_supply_depots=_num_complete(supply_depots),
			# Barracks
			num_barracks=num_barracks,
			num_completed_barracks=_num_complete(barracks),
        )

        return {
            **command_centers_state,
            **buildings_state
        }

    def _get_workers_state(self, obs: TimeStep) -> Dict[str, int|float]:
        workers = self.get_workers(obs)
        num_mineral_harvesters = len([w for w in workers if w.order_id_0 in Constants.HARVEST_ACTIONS])
        num_workers = len(workers)
        num_idle_workers = len([w for w in workers if self.is_idle(w)])
        pct_idle_workers = 0 if num_workers == 0 else num_idle_workers / num_workers
        pct_mineral_harvesters = 0 if num_workers == 0 else num_mineral_harvesters / num_workers

        # TODO more stats on N workers (e.g. distance to command centers, distance to minerals, to geysers...)
        return dict(
            num_workers=num_workers,
			num_idle_workers=len([w for w in workers if self.is_idle(w)]),
            pct_idle_workers=pct_idle_workers,
            num_mineral_harvesters=num_mineral_harvesters,
            pct_mineral_harvesters=pct_mineral_harvesters,
        )

    def _get_army_state(self, obs: TimeStep) -> Dict[str, int|float]:
        marines = self.get_self_units(obs, unit_types=units.Terran.Marine)
        barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks)
        num_marines_in_queue = sum(map(lambda b: b.order_length, barracks))
        num_marines = len(marines)
        total_army_health = sum(map(lambda b: b.health, marines))

        return dict(
            num_marines=num_marines,
            num_marines_in_queue=num_marines_in_queue,
            total_army_health=total_army_health,
        )

    def _get_resources_state(self, obs: TimeStep) -> Dict[str, int|float]:
        return dict(
            free_supply=self.get_free_supply(obs),
            minerals=obs.observation.player.minerals,
            collection_rate_minerals=obs.observation.score_cumulative.collection_rate_minerals/60
        )

    def _get_scores_state(self, obs: TimeStep)     -> Dict[str, int|float]:
        return {
            "score_cumulative_score": obs.observation.score_cumulative.score,
            "score_cumulative_total_value_units": obs.observation.score_cumulative.total_value_units,
            "score_cumulative_total_value_structures": obs.observation.score_cumulative.total_value_structures,
            "score_cumulative_killed_value_units": obs.observation.score_cumulative.killed_value_units,
            "score_cumulative_killed_value_structures": obs.observation.score_cumulative.killed_value_structures,
            # Supply (food) scores
            "score_food_used_none": obs.observation.score_by_category.food_used.none,
            "score_food_used_army": obs.observation.score_by_category.food_used.army,
            "score_food_used_economy": obs.observation.score_by_category.food_used.economy,
            # Used minerals and vespene
            "score_used_minerals_none": obs.observation.score_by_category.used_minerals.none,
            "score_used_minerals_army": obs.observation.score_by_category.used_minerals.army,
            "score_used_minerals_economy": obs.observation.score_by_category.used_minerals.economy,
            # Score by vital
            "score_by_vital_total_damage_dealt_life": obs.observation.score_by_vital.total_damage_dealt.life,
            "score_by_vital_total_damage_taken_life": obs.observation.score_by_vital.total_damage_taken.life,
        }

    def _get_neutral_units_state(self, obs: TimeStep) -> Dict[str, int|float]:
        minerals = [unit.tag for unit in obs.observation.raw_units if Minerals.contains(unit.unit_type)]
        return dict(
            num_minerals=len(minerals),
        )

    def _get_enemy_state(self, obs: TimeStep) -> Dict[str, int|float]:
        enemy_buildings = self.get_enemy_units(obs, unit_types=Constants.BUILDING_UNIT_TYPES)
        enemy_army = self.get_enemy_units(obs, unit_types=Constants.ARMY_UNIT_TYPES)

        return dict(
            enemy_total_building_health=sum(map(lambda b: b.health, enemy_buildings)),
            enemy_total_army_health=sum(map(lambda b: b.health, enemy_army)),
        )

    def _get_actions_state(self, obs: TimeStep) -> Dict[str, int]:
        available_actions = self.available_actions(obs)
        return dict(
            can_harvest_minerals=int(AllActions.HARVEST_MINERALS in available_actions),
            can_recruit_worker=int(AllActions.RECRUIT_SCV in available_actions),
            can_build_supply_depot=int(AllActions.BUILD_SUPPLY_DEPOT in available_actions),
            can_build_command_center=int(AllActions.BUILD_COMMAND_CENTER in available_actions),
            can_build_barracks=int(AllActions.BUILD_BARRACKS in available_actions),
            can_recruit_marine=int(AllActions.RECRUIT_MARINE in available_actions),
            can_attack=int(AllActions.ATTACK_WITH_SINGLE_UNIT in available_actions),
        )