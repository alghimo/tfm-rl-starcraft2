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

# from ..with_tracker import WithTracker
from .stats import AgentStats, AggregatedEpisodeStats, EpisodeStats

# from .agent_utils import AgentUtils


class BaseAgent(WithLogger, ABC, base_agent.BaseAgent):
    _AGENT_FILE: str = "agent.pkl"
    _BUFFER_FILE: str = "buffer.pkl"
    _STATS_FILE: str =  "stats.parquet"

    _action_to_game = {
        AllActions.NO_OP: actions.RAW_FUNCTIONS.no_op,
        AllActions.HARVEST_MINERALS: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", source_unit_tag, target_unit_tag),
        # AllActions.COLLECT_GAS: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", source_unit_tag, target_unit_tag),
        # AllActions.BUILD_REFINERY: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Build_Refinery_pt("now", source_unit_tag, target_unit_tag),
        AllActions.RECRUIT_SCV: lambda source_unit_tag: actions.RAW_FUNCTIONS.Train_SCV_quick("now", source_unit_tag),
        AllActions.BUILD_SUPPLY_DEPOT: lambda source_unit_tag, target_position: actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", source_unit_tag, target_position),
        AllActions.BUILD_COMMAND_CENTER: lambda source_unit_tag, target_position: actions.RAW_FUNCTIONS.Build_CommandCenter_pt("now", source_unit_tag, target_position),
        AllActions.BUILD_BARRACKS: lambda source_unit_tag, target_position: actions.RAW_FUNCTIONS.Build_Barracks_pt("now", source_unit_tag, target_position),
        AllActions.RECRUIT_MARINE: lambda source_unit_tag: actions.RAW_FUNCTIONS.Train_Marine_quick("now", source_unit_tag),
        AllActions.ATTACK_WITH_SINGLE_UNIT: lambda source_unit_tag, target_position: actions.RAW_FUNCTIONS.Attack_pt("now", source_unit_tag, target_position),
        # AllActions.ATTACK_WITH_SINGLE_UNIT: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tag, target_unit_tag),
        # AllActions.ATTACK_WITH_SQUAD_5: lambda source_unit_tags, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tags, target_unit_tag),
        # AllActions.ATTACK_WITH_SQUAD_10: lambda source_unit_tags, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tags, target_unit_tag),
        # AllActions.ATTACK_WITH_SQUAD_15: lambda source_unit_tags, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tags, target_unit_tag),
        # AllActions.ATTACK_WITH_FULL_ARMY: lambda source_unit_tags, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tags, target_unit_tag)
    }

    def __init__(self,
                 map_name: str, map_config: Dict, train: bool = True, checkpoint_path: Union[str|Path] = None,
                 tracker_update_freq_seconds: int = 10, reward_method: RewardMethod = RewardMethod.REWARD,
                 buffer: ExperienceReplayBuffer = None, **kwargs):
        super().__init__(**kwargs)
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
        self._current_state = None
        self._prev_state = None

        self._action_to_idx = None
        self._idx_to_action = None
        self._num_actions = None

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

        self.save_stats(self.checkpoint_path)

        if self._buffer is not None:
            with open(self._buffer_path, "wb") as f:
                pickle.dump(self._buffer, f)

    @classmethod
    def load(cls, checkpoint_path: Union[str|Path], map_name: str, map_config: Dict) -> Self:
        checkpoint_path = Path(checkpoint_path)
        agent_attrs_file = checkpoint_path / cls._AGENT_FILE
        with open(agent_attrs_file, mode="rb") as f:
            agent_attrs = pickle.load(f)

        init_attrs = cls._extract_init_arguments(agent_attrs=agent_attrs, map_name=map_name, map_config=map_config)
        agent = cls(**init_attrs)
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

        try:
            all_episode_stats = [v for v in self._episode_stats.values()]
            all_episode_stats = reduce(lambda v1, v2: v1 + v2, all_episode_stats)
            episode_stats_pd = pd.DataFrame(data=all_episode_stats)
            episode_stats_pd.to_parquet(self._checkpoint_path / "episode_stats.parquet")
            self.logger.info(f"Saved episode stats")
        except Exception as error:
            self.logger.error(f"Error saving episode stats")
            self.logger.exception(error)

        try:
            all_agent_stats = [v for v in self._agent_stats.values()]
            agent_stats_pd = pd.DataFrame(data=all_agent_stats)
            agent_stats_pd.to_parquet(self._checkpoint_path / "agent_stats.parquet")
            self.logger.info(f"Saved agent stats")
        except Exception as error:
            self.logger.error(f"Error saving agent stats")
            self.logger.exception(error)

        try:
            all_aggregated_stats = [v for v in self._aggregated_episode_stats.values()]
            aggregated_stats_pd = pd.DataFrame(data=all_aggregated_stats)
            aggregated_stats_pd.to_parquet(self._checkpoint_path / "aggregated_stats.parquet")
            self.logger.info(f"Saved aggregated stats")
        except Exception as error:
            self.logger.error(f"Error saving aggregated stats")
            self.logger.exception(error)

    @classmethod
    def _extract_init_arguments(cls, agent_attrs: Dict[str, Any], map_name: str, map_config: Dict) -> Dict[str, Any]:
        return dict(
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
        self._agent_path = agent_attrs["agent_path"]
        self._stats_path = agent_attrs["stats_path"]
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
        self._prev_state = None
        self._current_state = None

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
                    # closest_worker, closest_mineral = self.select_closest_worker(obs, idle_workers, command_centers, minerals)
                    action_args = dict(source_unit_tag=worker.tag, target_unit_tag=mineral.tag)
                # case AllActions.BUILD_REFINERY:
                #     geysers = [unit for unit in obs.observation.raw_units if Gas.contains(unit.unit_type)]
                #     command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
                #     workers = self.get_idle_workers(obs)
                #     if len(workers) == 0:
                #         workers = self.get_harvester_workers(obs)

                #     if len(command_centers) == 0:
                #         # Only takes into account distance between workers and geysers
                #         closest_worker, closest_geyser = self.select_closest_worker_to_resource(obs, workers, geysers)
                #     else:
                #         # Takes into account the distance from the worker to the command center and from the command center to the geyser
                #         closest_worker, closest_geyser = self.select_closest_worker(obs, workers, command_centers, geysers)
                #     action_args = dict(source_unit_tag=closest_worker.tag, target_unit_tag=closest_geyser.tag)
                # case AllActions.COLLECT_GAS:
                #     refineries = self.get_self_units(obs, unit_types=[units.Terran.Refinery, units.Terran.RefineryRich])
                #     command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
                #     idle_workers = self.get_idle_workers(obs)

                #     closest_worker, closest_refinery = self.select_closest_worker(obs, idle_workers, command_centers, refineries)
                #     action_args = dict(source_unit_tag=closest_worker.tag, target_unit_tag=closest_refinery.tag)
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
                # case AllActions.ATTACK_WITH_SINGLE_UNIT:
                #     idle_marines = self.get_idle_marines(obs)
                #     enemies = self.get_enemy_units(obs)
                #     marine_tag = random.choice(idle_marines).tag
                #     enemy_tag = random.choice(enemies).tag
                #     action_args = dict(source_unit_tag=marine_tag, target_unit_tag=enemy_tag)
                # case AllActions.ATTACK_WITH_SQUAD_5:
                #     idle_marines = [m.tag for m in self.get_idle_marines(obs)]
                #     enemies = self.get_enemy_units(obs)
                #     enemy_tag = random.choice(enemies).tag
                #     marine_tags = random.sample(idle_marines, k=5)
                #     action_args = dict(source_unit_tags=marine_tags, target_unit_tag=enemy_tag)
                # case AllActions.ATTACK_WITH_SQUAD_10:
                #     idle_marines = [m.tag for m in self.get_idle_marines(obs)]
                #     enemies = self.get_enemy_units(obs)
                #     enemy_tag = random.choice(enemies).tag
                #     marine_tags = random.sample(idle_marines, k=10)
                #     action_args = dict(source_unit_tags=marine_tags, target_unit_tag=enemy_tag)
                # case AllActions.ATTACK_WITH_SQUAD_15:
                #     idle_marines = [m.tag for m in self.get_idle_marines(obs)]
                #     enemies = self.get_enemy_units(obs)
                #     enemy_tag = random.choice(enemies).tag
                #     marine_tags = random.sample(idle_marines, k=15)
                #     action_args = dict(source_unit_tags=marine_tags, target_unit_tag=enemy_tag)
                # case AllActions.ATTACK_WITH_FULL_ARMY:
                #     idle_marines = [m.tag for m in self.get_idle_marines(obs)]
                #     assert len(idle_marines) > 0, "Can't attack with no units"
                #     enemies = self.get_enemy_units(obs)
                #     enemy_tag = random.choice(enemies).tag
                #     action_args = dict(source_unit_tags=idle_marines, target_unit_tag=enemy_tag)
                case _:
                    raise RuntimeError(f"Missing logic to select action args for action {action}")

            return action_args, True
        except AssertionError as error:
            self.logger.debug(error)
            return None, False

    def get_next_command_center_position(self, obs: TimeStep) -> Position:
        next_pos = None
        for idx, candidate_position in enumerate(self._command_center_positions):
            if candidate_position not in self._used_command_center_positions:
                next_pos = candidate_position
                break

        # if next_pos is not None:
        #     # put all positions before the candidate position at the end
        #     self._command_center_positions = self._command_center_positions[idx + 1:] + self._command_center_positions[:idx+1]

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

        # if next_pos is not None:
        #     # put all positions before the candidate position at the end
        #     self._supply_depot_positions = self._supply_depot_positions[idx + 1:] + self._supply_depot_positions[:idx+1]

        return next_pos

    def update_supply_depot_positions(self, obs: TimeStep) -> Position:
        supply_depots = self.get_self_units(obs, unit_types=units.Terran.SupplyDepot)
        supply_depots_positions = [Position(sd.x, sd.y) for sd in supply_depots]
        self._used_supply_depot_positions = supply_depots_positions
        # self._supply_depot_positions = [pos for pos in self._supply_depot_positions if pos not in supply_depots_positions]

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
        # if next_pos is not None:
        #     # put all positions before the candidate position at the end
        #     self._barrack_positions = self._barrack_positions[idx + 1:] + self._barrack_positions[:idx+1]

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
        adjusted_reward = reward + Constants.STEP_REWARD
        score_delta = obs.observation.score_cumulative.score - self._prev_score
        if not (obs.last() or obs.first()):
            if self._prev_action_is_valid == False:
                adjusted_reward = Constants.INVALID_ACTION_REWARD
                score_delta = Constants.INVALID_ACTION_REWARD
            elif self._prev_action == AllActions.NO_OP:
                adjusted_reward = Constants.NO_OP_REWARD
                score_delta = Constants.NO_OP_REWARD
        self._current_score = score_delta
        self._current_reward = reward
        self._current_adjusted_reward = adjusted_reward

        return reward, adjusted_reward, score_delta

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

        return torch.Tensor(State(
            **actions_state,
			**building_state,
			**worker_state,
			**army_state,
			**resources_state,
            **scores_state,
            **neutral_units_state,
            **enemy_state
        )).to(device=self.device)

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
        self._current_state = self._convert_obs_to_state(obs)
        reward, adjusted_reward, score = self.get_reward_and_score(obs)
        self._current_episode_stats.reward += reward
        self._current_episode_stats.adjusted_reward += adjusted_reward
        self._current_episode_stats.score += score
        self._current_episode_stats.steps += 1
        self.current_agent_stats.step_count += 1

        if obs.first():
            self._idx_to_action = { idx: action for idx, action in enumerate(self.agent_actions) }
            self._action_to_idx = { action: idx for idx, action in enumerate(self.agent_actions) }
            self._num_actions = len(self.agent_actions)
        else:
            done = obs.last()
            self._buffer.append(self._prev_state, self._prev_action, self._prev_action_args, self._current_reward, self._current_adjusted_reward, self._current_score, done, self._current_state)

    def post_step(self, obs: TimeStep, action: AllActions, action_args: Dict[str, Any], original_action: AllActions, original_action_args: Dict[str, Any], is_valid_action: bool):
        self._prev_score = obs.observation.score_cumulative.score
        self._prev_state = self._current_state
        self._prev_action = self._action_to_idx[original_action]
        self._prev_action_args = original_action_args

        if obs.last():
            emissions = self._tracker.flush() if self._tracker is not None else 0.
            self.logger.debug(f"End of episode - Got extra {emissions} since last update")
            self._current_episode_stats.emissions += emissions
            self._current_episode_stats.is_training = self.is_training
            self._current_episode_stats.is_exploit = self._exploit
            self._current_episode_stats.final_stage = self._current_agent_stage().name
            self._tracker_last_update = time.time()
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
        current_stage = self._current_agent_stage()
        mean_rewards = self.current_aggregated_episode_stats.mean_rewards(stage=current_stage, reward_method=RewardMethod.REWARD)
        mean_rewards_10 = self.current_aggregated_episode_stats.mean_rewards(stage=current_stage, last_n=10, reward_method=RewardMethod.REWARD)
        mean_adjusted_rewards = self.current_aggregated_episode_stats.mean_rewards(stage=current_stage, reward_method=RewardMethod.ADJUSTED_REWARD)
        mean_adjusted_rewards_10 = self.current_aggregated_episode_stats.mean_rewards(stage=current_stage, last_n=10, reward_method=RewardMethod.ADJUSTED_REWARD)
        mean_scores = self.current_aggregated_episode_stats.mean_rewards(stage=current_stage, reward_method=RewardMethod.SCORE)
        mean_scores_10 = self.current_aggregated_episode_stats.mean_rewards(stage=current_stage, last_n=10, reward_method=RewardMethod.SCORE)
        episode_count = self.current_agent_stats.episode_count_per_stage[current_stage.name]
        return [
            f"Stage: {self._current_agent_stage().name}",
            f"Reward method: {self._reward_method.name}",
            f"Episode {episode_count}",
            f"Mean Rewards for stage ({episode_count} ep) {mean_rewards:.2f} / (10ep) {mean_rewards_10:.2f}",
            f"Mean Adjusted Rewards for stage ({episode_count} ep) {mean_adjusted_rewards:.2f} / (10ep) {mean_adjusted_rewards_10:.2f}",
            f"Mean Scores ({episode_count} ep) {mean_scores:.2f} / (10ep) {mean_scores_10:.2f}",
            f"Episode steps: {self._current_episode_stats.steps} / Total steps: {self.current_agent_stats.step_count_per_stage[current_stage.name]}",
            f"Invalid actions: {num_invalid}/{num_valid + num_invalid} ({100 * pct_invalid:.2f}%)",
            f"Max reward {self.current_aggregated_episode_stats.max_reward_per_stage[current_stage.name]:.2f} (absolute max: {self.current_aggregated_episode_stats.max_reward})",
            f"Max adjusted reward {self.current_aggregated_episode_stats.max_adjusted_reward_per_stage[current_stage.name]:.2f} (absolute max: {self.current_aggregated_episode_stats.max_adjusted_reward})",
            f"Max score {self.current_aggregated_episode_stats.max_score_per_stage[current_stage.name]:.2f} (absolute max: {self.current_aggregated_episode_stats.max_score})",
            f"Episode emissions: {self._current_episode_stats.emissions} / Total in stage: {self.current_agent_stats.total_emissions_per_stage[current_stage.name]} / Total: {self.current_agent_stats.total_emissions}",
        ]

    def step(self, obs: TimeStep) -> AllActions:
        if obs.first():
            self._setup_positions(obs)
        self.pre_step(obs)

        super().step(obs)

        # if not obs.first():
        #     import pdb
        #     pdb.set_trace()
        self.update_supply_depot_positions(obs)
        self.update_command_center_positions(obs)
        self.update_barracks_positions(obs)

        action, action_args, is_valid_action = self.select_action(obs)
        original_action = action
        original_action_args = action_args
        # if not is_valid_action and (action == AllActions.ATTACK_WITH_SINGLE_UNIT) and self.can_take(obs, AllActions.RECRUIT_MARINE):
        #     self.logger.warning(f"Converting action {action.name} to RECRUIT_MARINE")
        #     action = AllActions.RECRUIT_MARINE
        #     action_args, is_valid_action = self._get_action_args(obs, action)
        # if not is_valid_action and (action == AllActions.RECRUIT_MARINE) and self.can_take(obs, AllActions.BUILD_BARRACKS):
        #     self.logger.warning(f"Converting action {action.name} to BUILD_BARRACKS")
        #     action = AllActions.BUILD_BARRACKS
        #     action_args, is_valid_action = self._get_action_args(obs, action)

        if not is_valid_action:
            self.logger.debug(f"Action {action.name} is not valid anymore, returning NO_OP")
            action = AllActions.NO_OP
            action_args = None
            self._current_episode_stats.add_invalid_action(action)
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
            self.logger.warning(f"Tried to validate action {action.name} that is not available for this agent. Allowed actions: {self.agent_actions}")
            return False
        elif action not in self._map_config["available_actions"]:
            # self.logger.debug(f"Tried to validate action {action.name} that is not available for this map.")
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
            # case AllActions.HARVEST_MINERALS, args if _has_target_unit_tag(args) and _has_source_unit_tag(args):
            #     self.logger.debug(f"Checking action {action.name} ({action}) with source and target units")
            #     command_centers = [unit.tag for unit in self.get_self_units(obs, unit_types=units.Terran.CommandCenter)]
            #     if not any(command_centers):
            #         self.logger.debug(f"[Action {action.name} ({action})] The player has no command centers")
            #         return False

            #     minerals = [unit.tag for unit in obs.observation.raw_units if Minerals.contains(unit.unit_type) and unit.x == position.x and unit.y == position.y]
            #     if not any(minerals):
            #         self.logger.debug(f"[Action {action.name} ({action}) + position] There are no minerals to harvest at position {position}")
            #         return False

            #     if self.has_idle_workers(obs):
            #         return True
            #     # TODO Add check for workers harvesting gas

            #     self.logger.debug(f"[Action {action.name} ({action}) + position] The target position has minerals, but the player has no SCVs.")
            #     return False
            case AllActions.HARVEST_MINERALS, args if _has_target_unit_tag(args):
                target_unit_tag = args["target_unit_tag"]
                self.logger.debug(f"Checking action {action.name} ({action}) with position")
                command_centers = [unit.tag for unit in self.get_self_units(obs, unit_types=units.Terran.CommandCenter)]
                if not any(command_centers):
                    self.logger.debug(f"[Action {action.name} ({action}) + position] The player has no command centers")
                    return False

                minerals = [unit.tag for unit in obs.observation.raw_units if unit.tag == target_unit_tag and Minerals.contains(unit.unit_type)]
                if len(minerals) == 0:
                    self.logger.debug(f"[Action {action.name} ({action}) + position] There are no minerals to harvest at position {position}")
                    return False

                if self.has_idle_workers(obs):
                    return True
                # TODO Add check for workers harvesting gas

                self.logger.debug(f"[Action {action.name} ({action}) + position] The target position has minerals, but the player has no SCVs.")
                return False
            case AllActions.HARVEST_MINERALS, _:
                self.logger.debug(f"Checking action {action.name} ({action}) with no position")
                command_centers = [unit.tag for unit in self.get_self_units(obs, unit_types=units.Terran.CommandCenter)]

                if not any(command_centers):
                    self.logger.debug(f"[Action {action.name} ({action}) + position] The player has no command centers")
                    return False

                minerals = [unit.tag for unit in obs.observation.raw_units if Minerals.contains(unit.unit_type)]
                if not any(minerals):
                    self.logger.debug(f"[Action {action.name} ({action}) without position] There are no minerals on the map")
                    return False

                if self.has_idle_workers(obs):
                    return True
                # elif self.has_workers(obs):
                #     return True
                # TODO Add check for workers harvesting gas

                self.logger.debug(f"[Action {action.name} ({action}) without position] There are minerals available, but the player has no SCVs.")
                return False
            # case AllActions.COLLECT_GAS, args if _has_target_unit_tag(args):
            #     target_unit_tag = args["target_unit_tag"]
            #     self.logger.debug(f"Checking action {action.name} ({action}) with position")
            #     command_centers = [unit.tag for unit in self.get_self_units(obs, unit_types=units.Terran.CommandCenter)]
            #     if not any(command_centers):
            #         self.logger.debug(f"[Action {action.name} ({action}) + position] The player has no command centers")
            #         return False

            #     refineries = self.get_self_units(obs, unit_types=[units.Terran.Refinery, units.Terran.RefineryRich], unit_tags=target_unit_tag)

            #     if len(refineries) == 0:
            #         self.logger.debug(f"[Action {action.name} ({action}) + position] There are no refineries to harvest at position {position}")
            #         return False

            #     if self.has_idle_workers(obs):
            #         return True
            #     # TODO Add check for workers harvesting minerals

            #     self.logger.debug(f"[Action {action.name} ({action}) + position] The target position has minerals, but the player has no SCVs.")
            #     return False
            # case AllActions.COLLECT_GAS, _:
            #     self.logger.debug(f"Checking action {action.name} ({action}) with no position")
            #     command_centers = [unit.tag for unit in self.get_self_units(obs, unit_types=units.Terran.CommandCenter)]
            #     if not any(command_centers):
            #         self.logger.debug(f"[Action {action.name} ({action}) + position] The player has no command centers")
            #         return False

            #     refineries = self.get_self_units(obs, unit_types=[units.Terran.Refinery, units.Terran.RefineryRich])
            #     refineries = [unit.tag for unit in refineries]
            #     if not any(refineries):
            #         self.logger.debug(f"[Action {action.name} ({action}) without position] There are no refineries on the map")
            #         return False

            #     if self.has_idle_workers(obs):
            #         return True
            #     # elif self.has_workers(obs):
            #     #     return True
            #     # TODO Add check for workers harvesting minerals

            #     self.logger.debug(f"[Action {action.name} ({action}) without position] There are refineries, but the player has no workers.")
            #     return False
            # TODO Add this check
            # case AllActions.BUILD_REFINERY, args if _has_source_unit_tag(args) and _has_target_unit_tag(args):
            # case AllActions.BUILD_REFINERY, args if _has_target_unit_tag(args):
            #     target_unit_tag = args["target_unit_tag"]
            #     geysers = [unit.tag for unit in obs.observation.raw_units if unit.tag == target_unit_tag and Gas.contains(unit.unit_type)]
            #     if not any(geysers):
            #         self.logger.debug(f"[Action {action.name} ({action}) + position] There are no vespene geysers at position {position} (or they already have a structure)")
            #         return False

            #     if not SC2Costs.REFINERY.can_pay(obs.observation.player):
            #         self.logger.debug(f"[Action {action.name} ({action}) + position] There are is a vespene geyser at position {position} but the player can't pay the cost ({SC2Costs.REFINERY})")
            #         return False

            #     if self.has_idle_workers(obs):
            #         return True
            #     elif self.has_harvester_workers(obs):
            #         self.logger.debug(f"[Action {action.name} ({action}) + position] Player has no idle SCVs, but has other available workers harvesting.")
            #         return True

            #     self.logger.debug(f"[Action {action.name} ({action}) + position] The target position has a vespene geyser and the player can pay the cust, but the player has no SCVs.")
            #     return False
            # case AllActions.BUILD_REFINERY, _:
            #     geysers = [unit.tag for unit in obs.observation.raw_units if Gas.contains(unit.unit_type)]
            #     if not any(geysers):
            #         self.logger.debug(f"[Action {action.name} ({action}) without position] There are no vespene geysers on the map(or they already have a structure)")
            #         return False

            #     if not SC2Costs.REFINERY.can_pay(obs.observation.player):
            #         self.logger.debug(f"[Action {action.name} ({action}) without position] There are are vespene geysers available but the player can't pay the cost ({SC2Costs.REFINERY})")
            #         return False

            #     if self.has_idle_workers(obs):
            #         return True
            #     elif self.has_workers(obs):
            #         self.logger.debug(f"[Action {action.name} ({action}) without position] Player has no idle SCVs, but has other available SCVs.")
            #         return True

            #     self.logger.debug(f"[Action {action.name} ({action}) without position] There are free vespene geysers and the player can pay the cust, but the player has no SCVs.")
            #     return False
            case AllActions.RECRUIT_SCV, args if _has_source_unit_tag(args):
                command_center_tag = args["source_unit_tag"]
                command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter, unit_tags=command_center_tag)
                command_centers = [cc for cc in command_centers if cc.order_length < Constants.COMMAND_CENTER_QUEUE_LENGTH]
                if len(command_centers) == 0:
                    self.logger.debug(f"[Action {action.name} ({action})] The player has no command centers")
                    return False
                if command_centers[0].order_length >= Constants.COMMAND_CENTER_QUEUE_LENGTH:
                    self.logger.debug(f"[Action {action.name} ({action})] The command center has the build queue full")
                    return False
                if not SC2Costs.SCV.can_pay(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action})] The player can't pay the cost of an SCV ({SC2Costs.SCV})")
                    return False
                return True
            case AllActions.RECRUIT_SCV, _:
                self.logger.debug(f"Checking action {action.name} ({action}) with no command center tag")
                command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
                if len(command_centers) == 0:
                    self.logger.debug(f"[Action {action.name} ({action})] The player has no command centers")
                    return False

                for command_center in command_centers:
                    if self.can_take(obs, action, source_unit_tag=command_center.tag):
                        return True
            case AllActions.BUILD_SUPPLY_DEPOT, _:
                target_position = self.get_next_supply_depot_position(obs)
                if target_position is None:
                    self.logger.debug(f"[Action {action.name} ({action})] There are no free positions to build a supply depot")
                    return False
                if not self.has_idle_workers(obs):
                    if not self.has_harvester_workers(obs):
                        self.logger.debug(f"[Action {action.name} ({action})] Player has no idle workers or workers that are harvesting.")
                        return False
                    self.logger.debug(f"[Action {action.name} ({action})] Player has no idle workers, but has workers that are harvesting.")
                if not SC2Costs.SUPPLY_DEPOT.can_pay(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action})] The player can't pay the cost of a Supply Depot ({SC2Costs.SUPPLY_DEPOT})")
                    return False

                return True
            case AllActions.BUILD_COMMAND_CENTER, _:
                target_position = self.get_next_command_center_position(obs)
                if target_position is None:
                    self.logger.debug(f"[Action {action.name} ({action})] There are no free positions to build a command center")
                    return False

                if not self.has_idle_workers(obs):
                    if not self.has_harvester_workers(obs):
                        self.logger.debug(f"[Action {action.name} ({action})] Player has no idle or harvester workers.")
                        return False
                    self.logger.debug(f"[Action {action.name} ({action})] Player has no idle workers, but has workers that are harvesting.")

                if not SC2Costs.COMMAND_CENTER.can_pay(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action})] The player can't pay the cost of a Command Center ({SC2Costs.COMMAND_CENTER})")
                    return False

                return True
            case AllActions.BUILD_BARRACKS, _:
                target_position = self.get_next_barracks_position(obs)
                if target_position is None:
                    self.logger.debug(f"[Action {action.name} ({action})] There are no free positions to build barracks")
                    return False
                if not self.has_idle_workers(obs):
                    if not self.has_harvester_workers(obs):
                        self.logger.debug(f"[Action {action.name} ({action})] Player has no idle workers or workers that are harvesting.")
                        return False
                    self.logger.debug(f"[Action {action.name} ({action})] Player has no idle workers, but has workers that are harvesting.")
                if not SC2Costs.BARRACKS.can_pay(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action})] The player can't pay the cost of a Barrack ({SC2Costs.BARRACKS})")
                    return False

                return True
            case AllActions.RECRUIT_MARINE, args if _has_source_unit_tag(args):
                self.logger.debug(f"Checking action {action.name} ({action}) with barracks tag")
                barracks_tag = args["source_unit_tag"]
                barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks, unit_tags=barracks_tag)
                # TODO Review this if we ever add the option to build the reactor (queue size is increased to 8)
                barracks = [b for b in barracks if b.order_length < Constants.BARRACKS_QUEUE_LENGTH]
                if len(barracks) == 0:
                    self.logger.debug(f"[Action {action.name} ({action})] The player has no barracks with tag {barracks_tag}")
                    return False

                if not SC2Costs.MARINE.can_pay(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action})] The player can't pay the cost of a Marine ({SC2Costs.MARINE})")
                    return False
                return True

            case AllActions.RECRUIT_MARINE, _:
                self.logger.debug(f"Checking action {action.name} ({action}) with no barracks tag")
                barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks)
                # TODO Review this if we ever add the option to build the reactor (queue size is increased to 8)
                barracks = [b for b in barracks if b.order_length < Constants.BARRACKS_QUEUE_LENGTH]
                if len(barracks) == 0:
                    self.logger.debug(f"[Action {action.name} ({action})] The player has no barracks")
                    return False

                if not SC2Costs.MARINE.can_pay(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action})] The player can't pay the cost of a Marine ({SC2Costs.MARINE})")
                    return False

                return True
            case AllActions.ATTACK_WITH_SINGLE_UNIT, _:
                self.logger.debug(f"Checking action {action.name} ({action}) without source or target unit tags")
                idle_marines = self.get_idle_marines(obs)
                if len(idle_marines) == 0:
                    self.logger.debug(f"[Action {action.name} ({action})] The player has no idle marines")
                    return False
                enemies = self.get_enemy_units(obs)
                if len(enemies) == 0:
                    self.logger.debug(f"[Action {action.name} ({action})] The are no enemies to attack")
                    return False

                return True
            # case AllActions.ATTACK_WITH_SQUAD_5, _:
            #     self.logger.debug(f"Checking action {action.name} ({action}) without source or target unit tags")
            #     idle_marines = self.get_idle_marines(obs)
            #     if len(idle_marines) < 5:
            #         self.logger.debug(f"[Action {action.name} ({action})] The player has less than 5 idle marines")
            #         return False
            #     enemies = self.get_enemy_units(obs)
            #     if len(enemies) == 0:
            #         self.logger.debug(f"[Action {action.name} ({action})] The are no enemies to attack")
            #         return False

            #     return True
            # case AllActions.ATTACK_WITH_SQUAD_10, _:
            #     self.logger.debug(f"Checking action {action.name} ({action}) without source or target unit tags")
            #     idle_marines = self.get_idle_marines(obs)
            #     if len(idle_marines) < 10:
            #         self.logger.debug(f"[Action {action.name} ({action})] The player has less than 10 idle marines")
            #         return False
            #     enemies = self.get_enemy_units(obs)
            #     if len(enemies) == 0:
            #         self.logger.debug(f"[Action {action.name} ({action})] The are no enemies to attack")
            #         return False

            #     return True
            # case AllActions.ATTACK_WITH_SQUAD_15, _:
            #     self.logger.debug(f"Checking action {action.name} ({action}) without source or target unit tags")
            #     idle_marines = self.get_idle_marines(obs)
            #     if len(idle_marines) < 15:
            #         self.logger.debug(f"[Action {action.name} ({action})] The player has less than 15 idle marines")
            #         return False
            #     enemies = self.get_enemy_units(obs)
            #     if len(enemies) == 0:
            #         self.logger.debug(f"[Action {action.name} ({action})] The are no enemies to attack")
            #         return False

            #     return True
            # case AllActions.ATTACK_WITH_FULL_ARMY, _:
            #     self.logger.debug(f"Checking action {action.name} ({action}) without source or target unit tags")
            #     idle_marines = self.get_idle_marines(obs)
            #     if len(idle_marines) == 0:
            #         self.logger.debug(f"[Action {action.name} ({action})] The player has no idle marines")
            #         return False
            #     enemies = self.get_enemy_units(obs)
            #     if len(enemies) == 0:
            #         self.logger.debug(f"[Action {action.name} ({action})] The are no enemies to attack")
            #         return False

            #     return True
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

        for idx in range(4):
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
        refineries = self.get_self_units(obs, unit_types=units.Terran.Refinery)
        num_refineries = len(refineries)
        barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks)
        num_barracks = len(barracks)
        other_building_types = [
            bt for bt in Constants.BUILDING_UNIT_TYPES
            if bt not in [units.Terran.CommandCenter, units.Terran.SupplyDepot, units.Terran.Refinery, units.Terran.Barracks]
        ]
        other_buildings = self.get_self_units(obs, unit_types=other_building_types)

        other_buildings = [b for b in other_buildings if b not in []]
        num_other_buildings = len(other_buildings)
        buildings_state = dict(
			num_refineries=num_refineries,
			num_completed_refineries=_num_complete(refineries),
			# Supply Depots
			num_supply_depots=num_supply_depots,
			num_completed_supply_depots=_num_complete(supply_depots),
			# Barracks
			num_barracks=num_barracks,
			num_completed_barracks=_num_complete(barracks),
            num_other_buildings=num_other_buildings
        )

        return {
            **command_centers_state,
            **buildings_state
        }

    def _get_workers_state(self, obs: TimeStep) -> Dict[str, int|float]:
        workers = self.get_workers(obs)
        num_harvesters = len([w for w in workers if w.order_id_0 in Constants.HARVEST_ACTIONS])
        refineries = self.get_self_units(obs, unit_types=units.Terran.Refinery)
        num_gas_harvesters = sum(map(lambda r: r.assigned_harvesters, refineries))
        num_mineral_harvesters = num_harvesters - num_gas_harvesters
        num_workers = len(workers)
        num_idle_workers = len([w for w in workers if self.is_idle(w)])
        pct_idle_workers = 0 if num_workers == 0 else num_idle_workers / num_workers
        pct_mineral_harvesters = 0 if num_workers == 0 else num_mineral_harvesters / num_workers
        pct_gas_harvesters = 0 if num_workers == 0 else num_gas_harvesters / num_workers

        # TODO more stats on N workers (e.g. distance to command centers, distance to minerals, to geysers...)
        return dict(
            num_workers=num_workers,
			num_idle_workers=len([w for w in workers if self.is_idle(w)]),
            pct_idle_workers=pct_idle_workers,
            num_mineral_harvesters=num_mineral_harvesters,
            pct_mineral_harvesters=pct_mineral_harvesters,
            num_gas_harvesters=num_gas_harvesters,
            pct_gas_harvesters=pct_gas_harvesters
        )

    def _get_army_state(self, obs: TimeStep) -> Dict[str, int|float]:
        marines = self.get_self_units(obs, unit_types=units.Terran.Marine)
        barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks)
        num_marines_in_queue = sum(map(lambda b: b.order_length, barracks))
        other_army_units_types = [ut for ut in Constants.ARMY_UNIT_TYPES if ut not in [units.Terran.Marine]]
        other_army_units = self.get_self_units(obs, unit_types=other_army_units_types)
        return dict(
            num_marines=len(marines),
            num_marines_in_queue=num_marines_in_queue,
            num_other_army_units=len(other_army_units)
        )

    def _get_resources_state(self, obs: TimeStep) -> Dict[str, int|float]:
        return dict(
            free_supply=self.get_free_supply(obs),
            minerals=obs.observation.player.minerals,
            gas=obs.observation.player.vespene,
        )

    def _get_scores_state(self, obs: TimeStep)     -> Dict[str, int|float]:
        return {
            "score_cumulative_score": obs.observation.score_cumulative.score,
            "score_cumulative_idle_production_time": obs.observation.score_cumulative.idle_production_time,
            "score_cumulative_idle_worker_time": obs.observation.score_cumulative.idle_worker_time,
            "score_cumulative_total_value_units": obs.observation.score_cumulative.total_value_units,
            "score_cumulative_total_value_structures": obs.observation.score_cumulative.total_value_structures,
            "score_cumulative_killed_value_units": obs.observation.score_cumulative.killed_value_units,
            "score_cumulative_killed_value_structures": obs.observation.score_cumulative.killed_value_structures,
            "score_cumulative_collected_minerals": obs.observation.score_cumulative.collected_minerals,
            "score_cumulative_collected_vespene": obs.observation.score_cumulative.collected_vespene,
            "score_cumulative_collection_rate_minerals": obs.observation.score_cumulative.collection_rate_minerals,
            "score_cumulative_collection_rate_vespene": obs.observation.score_cumulative.collection_rate_vespene,
            "score_cumulative_spent_minerals": obs.observation.score_cumulative.spent_minerals,
            "score_cumulative_spent_vespene": obs.observation.score_cumulative.spent_vespene,
            # Supply (food) scores
            "score_food_used_none": obs.observation.score_by_category.food_used.none,
            "score_food_used_army": obs.observation.score_by_category.food_used.army,
            "score_food_used_economy": obs.observation.score_by_category.food_used.economy,
            "score_food_used_technology": obs.observation.score_by_category.food_used.technology,
            "score_food_used_upgrade": obs.observation.score_by_category.food_used.upgrade,
            # Killed minerals and vespene
            "score_killed_minerals_none": obs.observation.score_by_category.killed_minerals.none,
            "score_killed_minerals_army": obs.observation.score_by_category.killed_minerals.army,
            "score_killed_minerals_economy": obs.observation.score_by_category.killed_minerals.economy,
            "score_killed_minerals_technology": obs.observation.score_by_category.killed_minerals.technology,
            "score_killed_minerals_upgrade": obs.observation.score_by_category.killed_minerals.upgrade,
            "score_killed_vespene_none": obs.observation.score_by_category.killed_vespene.none,
            "score_killed_vespene_army": obs.observation.score_by_category.killed_vespene.army,
            "score_killed_vespene_economy": obs.observation.score_by_category.killed_vespene.economy,
            "score_killed_vespene_technology": obs.observation.score_by_category.killed_vespene.technology,
            "score_killed_vespene_upgrade": obs.observation.score_by_category.killed_vespene.upgrade,
            # Lost minerals and vespene
            "score_lost_minerals_none": obs.observation.score_by_category.lost_minerals.none,
            "score_lost_minerals_army": obs.observation.score_by_category.lost_minerals.army,
            "score_lost_minerals_economy": obs.observation.score_by_category.lost_minerals.economy,
            "score_lost_minerals_technology": obs.observation.score_by_category.lost_minerals.technology,
            "score_lost_minerals_upgrade": obs.observation.score_by_category.lost_minerals.upgrade,
            "score_lost_vespene_none": obs.observation.score_by_category.lost_vespene.none,
            "score_lost_vespene_army": obs.observation.score_by_category.lost_vespene.army,
            "score_lost_vespene_economy": obs.observation.score_by_category.lost_vespene.economy,
            "score_lost_vespene_technology": obs.observation.score_by_category.lost_vespene.technology,
            "score_lost_vespene_upgrade": obs.observation.score_by_category.lost_vespene.upgrade,
            # Friendly fire minerals and vespene
            "score_friendly_fire_minerals_none": obs.observation.score_by_category.friendly_fire_minerals.none,
            "score_friendly_fire_minerals_army": obs.observation.score_by_category.friendly_fire_minerals.army,
            "score_friendly_fire_minerals_economy": obs.observation.score_by_category.friendly_fire_minerals.economy,
            "score_friendly_fire_minerals_technology": obs.observation.score_by_category.friendly_fire_minerals.technology,
            "score_friendly_fire_minerals_upgrade": obs.observation.score_by_category.friendly_fire_minerals.upgrade,
            "score_friendly_fire_vespene_none": obs.observation.score_by_category.friendly_fire_vespene.none,
            "score_friendly_fire_vespene_army": obs.observation.score_by_category.friendly_fire_vespene.army,
            "score_friendly_fire_vespene_economy": obs.observation.score_by_category.friendly_fire_vespene.economy,
            "score_friendly_fire_vespene_technology": obs.observation.score_by_category.friendly_fire_vespene.technology,
            "score_friendly_fire_vespene_upgrade": obs.observation.score_by_category.friendly_fire_vespene.upgrade,
            # Used minerals and vespene
            "score_used_minerals_none": obs.observation.score_by_category.used_minerals.none,
            "score_used_minerals_army": obs.observation.score_by_category.used_minerals.army,
            "score_used_minerals_economy": obs.observation.score_by_category.used_minerals.economy,
            "score_used_minerals_technology": obs.observation.score_by_category.used_minerals.technology,
            "score_used_minerals_upgrade": obs.observation.score_by_category.used_minerals.upgrade,
            "score_used_vespene_none": obs.observation.score_by_category.used_vespene.none,
            "score_used_vespene_army": obs.observation.score_by_category.used_vespene.army,
            "score_used_vespene_economy": obs.observation.score_by_category.used_vespene.economy,
            "score_used_vespene_technology": obs.observation.score_by_category.used_vespene.technology,
            "score_used_vespene_upgrade": obs.observation.score_by_category.used_vespene.upgrade,
            # Total used minerals and vespene
            "score_total_used_minerals_none": obs.observation.score_by_category.total_used_minerals.none,
            "score_total_used_minerals_army": obs.observation.score_by_category.total_used_minerals.army,
            "score_total_used_minerals_economy": obs.observation.score_by_category.total_used_minerals.economy,
            "score_total_used_minerals_technology": obs.observation.score_by_category.total_used_minerals.technology,
            "score_total_used_minerals_upgrade": obs.observation.score_by_category.total_used_minerals.upgrade,
            "score_total_used_vespene_none": obs.observation.score_by_category.total_used_vespene.none,
            "score_total_used_vespene_army": obs.observation.score_by_category.total_used_vespene.army,
            "score_total_used_vespene_economy": obs.observation.score_by_category.total_used_vespene.economy,
            "score_total_used_vespene_technology": obs.observation.score_by_category.total_used_vespene.technology,
            "score_total_used_vespene_upgrade": obs.observation.score_by_category.total_used_vespene.upgrade,

            # Score by vital
            "score_by_vital_total_damage_dealt_life": obs.observation.score_by_vital.total_damage_dealt.life,
            "score_by_vital_total_damage_dealt_shields": obs.observation.score_by_vital.total_damage_dealt.shields,
            "score_by_vital_total_damage_dealt_energy": obs.observation.score_by_vital.total_damage_dealt.energy,
            "score_by_vital_total_damage_taken_life": obs.observation.score_by_vital.total_damage_taken.life,
            "score_by_vital_total_damage_taken_shields": obs.observation.score_by_vital.total_damage_taken.shields,
            "score_by_vital_total_damage_taken_energy": obs.observation.score_by_vital.total_damage_taken.energy,
            "score_by_vital_total_healed_life": obs.observation.score_by_vital.total_healed.life,
            "score_by_vital_total_healed_shields": obs.observation.score_by_vital.total_healed.shields,
            "score_by_vital_total_healed_energy": obs.observation.score_by_vital.total_healed.energy,
        }

    def _get_neutral_units_state(self, obs: TimeStep) -> Dict[str, int|float]:
        minerals = [unit.tag for unit in obs.observation.raw_units if Minerals.contains(unit.unit_type)]
        geysers = [unit.tag for unit in obs.observation.raw_units if Gas.contains(unit.unit_type)]
        return dict(
            num_minerals=len(minerals),
            num_geysers=len(geysers),
        )

    def _get_enemy_state(self, obs: TimeStep) -> Dict[str, int|float]:
        enemy_buildings = self.get_enemy_units(obs, unit_types=Constants.BUILDING_UNIT_TYPES)
        enemy_workers = self.get_enemy_units(obs, unit_types=Constants.WORKER_UNIT_TYPES)
        enemy_army = self.get_enemy_units(obs, unit_types=Constants.ARMY_UNIT_TYPES)

        return dict(
            enemy_num_buildings=len(enemy_buildings),
            enemy_total_building_health=sum(map(lambda b: b.health, enemy_buildings)),
            enemy_num_workers=len(enemy_workers),
            enemy_num_army_units = len(enemy_army),
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