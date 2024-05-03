import pickle
from collections import namedtuple
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, units
from typing_extensions import Self

from ..actions import AllActions
from ..constants import Constants, SC2Costs
from ..networks.dqn_network import DQNNetwork
from ..networks.experience_replay_buffer import ExperienceReplayBuffer
from ..types import Gas, Minerals
from .base_agent import BaseAgent
from .stats import AgentStats, AggregatedEpisodeStats, EpisodeStats

DQNAgentParams = namedtuple('DQNAgentParams',
                            field_names=["epsilon", "epsilon_decay", "min_epsilon", "batch_size", "gamma", "loss", "main_network_update_frequency", "target_network_sync_frequency", "target_sync_mode", "update_tau"],
                            defaults=(0.1, 0.99, 0.01, 32, 0.99, None, 1, 50, "soft", 0.001))

State = namedtuple('State',
                            field_names=[
                                "can_harvest_minerals", "can_recruit_worker", "can_build_supply_depot", "can_build_command_center",
                                "can_build_barracks",  "can_recruit_marine",  "can_attack",
                                # Actions available on the map
                                # "map_actions",
                                # Command centers
                                "num_command_centers", "num_completed_command_centers",
                                "command_center_0_order_length", "command_center_1_order_length", "command_center_2_order_length", "command_center_3_order_length",
                                "command_center_0_num_workers", "command_center_1_num_workers", "command_center_2_num_workers", "command_center_3_num_workers",
                                # Workers
                                "num_workers", "num_idle_workers", "pct_idle_workers", "num_mineral_harvesters", "pct_mineral_harvesters", "num_gas_harvesters",  "pct_gas_harvesters",
                                # Buildings
                                "num_refineries", "num_completed_refineries", "num_supply_depots", "num_completed_supply_depots", "num_barracks", "num_completed_barracks", "num_other_buildings",
                                # Army
                                "num_marines", "num_marines_in_queue", "num_other_army_units",
                                # Resources
                                "free_supply", "minerals", "gas",
                                # Scores
                                # Cumulative
                                "score_cumulative_score", "score_cumulative_idle_production_time", "score_cumulative_idle_worker_time",
                                "score_cumulative_total_value_units", "score_cumulative_total_value_structures", "score_cumulative_killed_value_units",
                                "score_cumulative_killed_value_structures", "score_cumulative_collected_minerals", "score_cumulative_collected_vespene",
                                "score_cumulative_collection_rate_minerals", "score_cumulative_collection_rate_vespene", "score_cumulative_spent_minerals",
                                "score_cumulative_spent_vespene",
                                # By category
                                "score_food_used_none", "score_food_used_army", "score_food_used_economy", "score_food_used_technology", "score_food_used_upgrade",
                                "score_killed_minerals_none", "score_killed_minerals_army", "score_killed_minerals_economy", "score_killed_minerals_technology", "score_killed_minerals_upgrade",
                                "score_killed_vespene_none", "score_killed_vespene_army", "score_killed_vespene_economy", "score_killed_vespene_technology", "score_killed_vespene_upgrade",
                                "score_lost_minerals_none", "score_lost_minerals_army", "score_lost_minerals_economy", "score_lost_minerals_technology", "score_lost_minerals_upgrade",
                                "score_lost_vespene_none", "score_lost_vespene_army", "score_lost_vespene_economy", "score_lost_vespene_technology", "score_lost_vespene_upgrade",
                                "score_friendly_fire_minerals_none", "score_friendly_fire_minerals_army", "score_friendly_fire_minerals_economy", "score_friendly_fire_minerals_technology", "score_friendly_fire_minerals_upgrade",
                                "score_friendly_fire_vespene_none", "score_friendly_fire_vespene_army", "score_friendly_fire_vespene_economy", "score_friendly_fire_vespene_technology", "score_friendly_fire_vespene_upgrade",
                                "score_used_minerals_none", "score_used_minerals_army", "score_used_minerals_economy", "score_used_minerals_technology", "score_used_minerals_upgrade",
                                "score_used_vespene_none","score_used_vespene_army", "score_used_vespene_economy", "score_used_vespene_technology", "score_used_vespene_upgrade",
                                "score_total_used_minerals_none", "score_total_used_minerals_army", "score_total_used_minerals_economy", "score_total_used_minerals_technology", "score_total_used_minerals_upgrade",
                                "score_total_used_vespene_none", "score_total_used_vespene_army", "score_total_used_vespene_economy", "score_total_used_vespene_technology", "score_total_used_vespene_upgrade",
                                # Score by vital
                                "score_by_vital_total_damage_dealt_life", "score_by_vital_total_damage_dealt_shields", "score_by_vital_total_damage_dealt_energy",
                                "score_by_vital_total_damage_taken_life", "score_by_vital_total_damage_taken_shields", "score_by_vital_total_damage_taken_energy",
                                "score_by_vital_total_healed_life", "score_by_vital_total_healed_shields", "score_by_vital_total_healed_energy",
                                # Neutral units
                                "num_minerals", "num_geysers",
                                # Enemy info
                                "enemy_num_buildings", "enemy_total_building_health", "enemy_num_workers", "enemy_num_army_units", "enemy_total_army_health",
                            ])

class DQNAgent(BaseAgent):
    _MAIN_NETWORK_FILE: str = "main_network.pt"
    _TARGET_NETWORK_FILE: str = "target_network.pt"
    # _AGENT_FILE: str = "agent.pkl"
    # _STATS_FILE: str =  "stats.parquet"

    def __init__(self, main_network: DQNNetwork, buffer: ExperienceReplayBuffer,
                 hyperparams: DQNAgentParams,
                 target_network: DQNNetwork = None,
                 random_mode: bool = False,
                #  train: bool = True,
                #  checkpoint_path: Union[str|Path] = None,
                 **kwargs
                 ):
        """Deep Q-Network agent.

        Args:
            main_network (nn.Module): Main network
            buffer (ExperienceReplayBuffer): Memory buffer
            hyperparams (DQNAgentParams): Agent hyper parameters.
            target_network (nn.Module, optional): Target network. If not provided, then the main network will be cloned.
        """
        super().__init__(**kwargs)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.main_network = main_network
        self.target_network = target_network or deepcopy(main_network)
        self.buffer = buffer
        self.hyperparams = hyperparams
        self.initial_epsilon = hyperparams.epsilon
        self.epsilon = hyperparams.epsilon
        self.loss = self.hyperparams.loss or torch.nn.HuberLoss()
        self._random_mode = random_mode

        # Placeholders
        self._action_to_idx = None
        self._idx_to_action = None
        self._num_actions = None

        self.__current_state = None
        self.__prev_state = None
        self.__prev_reward = None
        self.__prev_action = None
        self.__prev_action_args = None

        # Add some extra flags to the default ones (train / exploit)
        self._status_flags.update(dict(
            burnin_started=False,
            main_net_updated=False,
            target_net_updated=False,
        ))

        if self._checkpoint_path is not None:
            self._main_network_path = self._checkpoint_path / self._MAIN_NETWORK_FILE
            self._target_network_path = self._checkpoint_path / self._TARGET_NETWORK_FILE
        else:
            self._main_network_path = None
            self._target_network_path = None

    def _update_checkpoint_paths(self):
        super()._update_checkpoint_paths()
        if self.checkpoint_path is None:
            self._main_network_path = None
            self._target_network_path = None
        else:
            self._main_network_path = self.checkpoint_path / self._MAIN_NETWORK_FILE
            self._target_network_path = self.checkpoint_path / self._TARGET_NETWORK_FILE

    @classmethod
    def _extract_init_arguments(cls, agent_attrs: Dict[str, Any], map_name: str, map_config: Dict) -> Dict[str, Any]:
        parent_attrs = super()._extract_init_arguments(agent_attrs=agent_attrs, map_name=map_name, map_config=map_config)
        return dict(
            **parent_attrs,
            main_network=torch.load(agent_attrs["main_network_path"]),
            target_network=torch.load(agent_attrs["target_network_path"]),
            buffer=agent_attrs["buffer"],
            random_mode=agent_attrs.get("random_mode", agent_attrs.get("random_model", False)),
            hyperparams=agent_attrs["hyperparams"]
        )

    def _load_agent_attrs(self, agent_attrs: Dict):
        super()._load_agent_attrs(agent_attrs)
        self._main_network_path = agent_attrs["main_network_path"]
        self._target_network_path = agent_attrs["target_network_path"]
        # self.device = agent_attrs["device"]
        self.initial_epsilon = agent_attrs["initial_epsilon"]
        # self._action_to_idx = agent_attrs["action_to_idx"]
        # self._idx_to_action = agent_attrs["idx_to_action"]
        self._num_actions = agent_attrs["num_actions"]
        # self.__current_state = agent_attrs["current_state"]
        # self.__prev_state = agent_attrs["prev_state"]
        # self.__prev_reward = agent_attrs["prev_reward"]
        # self.__prev_action = agent_attrs["prev_action"]
        # self.__prev_action_args = agent_attrs["prev_action_args"]
        self.loss = agent_attrs["loss"]

    def _get_agent_attrs(self):
        parent_attrs = super()._get_agent_attrs()
        return dict(
            buffer=self.buffer,
            hyperparams=self.hyperparams,
            initial_epsilon=self.initial_epsilon,
            # device=self.device,
            main_network_path=self._main_network_path,
            target_network_path=self._target_network_path,
            # action_to_idx=self._action_to_idx,
            # idx_to_action=self._idx_to_action,
            num_actions=self._num_actions,
            # current_state=self.__current_state,
            # prev_state=self.__prev_state,
            # prev_reward=self.__prev_reward,
            # prev_action=self.__prev_action,
            # prev_action_args=self.__prev_action_args,
            loss=self.loss,
            random_mode=self._random_mode,
            **parent_attrs
        )

    def save(self, checkpoint_path: Union[str|Path] = None):
        super().save(checkpoint_path=checkpoint_path)
        torch.save(self.main_network, self._main_network_path)
        torch.save(self.target_network, self._target_network_path)

    def reset(self, **kwargs):
        """Initialize the agent."""
        super().reset(**kwargs)
        # Last observation
        self.__prev_state = None
        self.__prev_reward = None
        self.__prev_action = None
        self.__prev_action_args = None
        self.__current_state = None

    @property
    def is_training(self):
        return super().is_training and (not self._random_mode) and (self.buffer.burn_in_capacity >= 1)

    def _convert_obs_to_state(self, obs: TimeStep) -> torch.Tensor:
        actions_state = self.get_actions_state(obs)
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


    def select_action(self, obs: TimeStep) -> Tuple[AllActions, Dict[str, Any]]:
        # available_actions = self.available_actions(obs)
        # self.logger.debug(f"Available actions: {available_actions}")
        available_actions = [a for a in self.agent_actions if a in self._map_config["available_actions"]]
        # if len(available_actions) > 1 and AllActions.NO_OP in available_actions:
        #     available_actions = [a for a in available_actions if a != AllActions.NO_OP]
        # One-hot encoded version of available actions
        valid_actions = self._actions_to_network(available_actions)
        if not any(valid_actions):
            valid_actions = None
        if (self._random_mode) or (self._train and (self.buffer.burn_in_capacity < 1)):
            if not self._status_flags["burnin_started"]:
                self.logger.info(f"Starting burn-in")
                self._status_flags["burnin_started"] = True
            if self._random_mode:
                self.logger.debug(f"Random mode - collecting experience from random actions")
            else:
                self.logger.debug(f"Burn in capacity: {100 * self.buffer.burn_in_capacity:.2f}%")
            raw_action = self.main_network.get_random_action(valid_actions=valid_actions)
            # raw_action = self.main_network.get_random_action()
        elif self.is_training:
            if not self._status_flags["train_started"]:
                self.logger.info(f"Starting training")
                self._status_flags["train_started"] = True
            raw_action = self.main_network.get_action(self.__current_state, epsilon=self.epsilon, valid_actions=valid_actions)
            # raw_action = self.main_network.get_action(self.__current_state, epsilon=self.epsilon)
        else:
            if not self._status_flags["exploit_started"]:
                self.logger.info(f"Starting exploit")
                self._status_flags["exploit_started"] = True

            available_actions = self.available_actions(obs)
            valid_actions = self._actions_to_network(available_actions)
            # available_actions = [a for a in self.agent_actions if a in self._map_config["available_actions"]]
            # One-hot encoded version of available actions
            # When exploiting, do not use the invalid action masking
            # raw_action = self.main_network.get_greedy_action(self.__current_state)
            raw_action = self.main_network.get_greedy_action(self.__current_state, valid_actions=valid_actions)

        # Convert the "raw" action to a the right type of action
        action = self._idx_to_action[raw_action]

        action_args, is_valid_action = self._get_action_args(obs=obs, action=action)

        return action, action_args, is_valid_action

    def pre_step(self, obs: TimeStep):
        self.__current_state = self._convert_obs_to_state(obs)
        reward = obs.reward

        self._current_episode_stats.reward += reward
        self._current_episode_stats.steps += 1
        self.current_agent_stats.step_count += 1

        if obs.first():
            self._idx_to_action = { idx: action for idx, action in enumerate(self.agent_actions) }
            self._action_to_idx = { action: idx for idx, action in enumerate(self.agent_actions) }
            self._num_actions = len(self.agent_actions)
            # self._map_actions = self._actions_to_network(self._map_config["available_actions"], as_tensor=False)
        else:
            # do updates
            done = obs.last()
            self.buffer.append(self.__prev_state, self.__prev_action, self.__prev_action_args, reward, done, self.__current_state)

            if self.is_training:
                updated = False
                if (self._current_episode_stats.steps % self.hyperparams.main_network_update_frequency) == 0:
                    if not self._status_flags["main_net_updated"]:
                        self.logger.info(f"First main network update")
                        self._status_flags["main_net_updated"] = True
                    self._current_episode_stats.losses = self.update_main_network(self._current_episode_stats.losses)
                    updated = True
                if (self._current_episode_stats.steps % self.hyperparams.target_network_sync_frequency) == 0:
                    if not self._status_flags["target_net_updated"]:
                        self.logger.info(f"First target network update")
                        self._status_flags["target_net_updated"] = True
                    self.synchronize_target_network()
                # HERE

                if done:
                    if not updated:
                        # If we finished but didn't update, perform one last update
                        self._current_episode_stats.losses = self.update_main_network(self._current_episode_stats.losses)

                    self._current_episode_stats.epsilon = self.epsilon if not self._random_mode else 1.
                    self.epsilon = max(self.epsilon * self.hyperparams.epsilon_decay, self.hyperparams.min_epsilon)
            elif self._exploit:
                self._current_episode_stats.epsilon = 0.

    def post_step(self, obs: TimeStep, action: AllActions, action_args: Dict[str, Any]):
        super().post_step(obs, action, action_args)

        if obs.first():
            self.__current_state = self._convert_obs_to_state(obs)
        elif obs.last():
            self._current_episode_stats.is_random_mode = self._random_mode

        self.__prev_state = self.__current_state
        self.__prev_action = self._action_to_idx[action]
        self.__prev_action_args = action_args

    def _get_end_of_episode_info_components(self) -> List[str]:
        return super()._get_end_of_episode_info_components() + [
            f"Epsilon {self._current_episode_stats.epsilon:.4f}",
            f"Mean Loss ({len(self._current_episode_stats.losses)} updates) {self._current_episode_stats.mean_loss:.4f}",
        ]

    def _calculate_loss(self, batch: Iterable[Tuple]) -> torch.Tensor:
        """Calculate the loss for a batch.

        Args:
            batch (Iterable[Tuple]): Batch to calculate the loss on.

        Returns:
            torch.Tensor: The calculated loss between the calculated and the predicted values.
        """

        # Convert elements from the replay buffer to tensors
        states, actions, action_args, rewards, dones, next_states = [i for i in batch]
        states = torch.stack(states).to(device=self.device)
        next_states = torch.stack(next_states).to(device=self.device)
        rewards_vals = torch.FloatTensor(rewards).to(device=self.device)
        actions_vals = torch.LongTensor(np.array(actions)).to(device=self.device).reshape(-1,1)
        dones_t = torch.BoolTensor(dones).to(device=self.device)

        # Get q-values from the main network
        qvals = torch.gather(self.main_network.get_qvals(states), 1, actions_vals)
        # Get q-values from the target network
        qvals_next = torch.max(self.target_network.get_qvals(next_states), dim=-1)[0].detach()
        # Set terminal states to 0
        qvals_next[dones_t] = 0

        # Get expected q-values
        expected_qvals = self.hyperparams.gamma * qvals_next + rewards_vals

        return self.loss(qvals, expected_qvals.reshape(-1,1))

    def update_main_network(self, episode_losses: List[float] = None) -> List[float]:
        """Update the main network.

        Normally we only perform the update every certain number of steps, defined by
        main_network_update_frequency, but if force_update is set to true, then
        the update will happen, independently of the current step count.

        Args:
            episode_losses (List, optional): List with episode losses.
        Returns:
            List[float]: A list with all the losses in the current episode.
        """
        episode_losses = episode_losses or []

        self.main_network.optimizer.zero_grad()  # Remove previous gradients
        batch = self.buffer.sample_batch(batch_size=self.hyperparams.batch_size) # Sample experiences from the buffer

        loss = self._calculate_loss(batch)# Get batch loss
        loss.backward() # Backward pass to get gradients
        self.main_network.optimizer.step() # Apply the gradients to the main network

        if self.device == 'cuda':
            loss = loss.detach().cpu().numpy()
        else:
            loss = loss.detach().numpy()

        episode_losses.append(float(loss))
        return episode_losses

    def synchronize_target_network(self):
        """Synchronize the target network with the main network parameters.

        When the target_sync_mode is set to "soft", a soft update is made, so instead of overwriting
        the target fully, we update it by mixing in the current target parameters and the main network
        parameters. In practice, we keep a fraction (1 - update_tau) from the target network, and add
        to it a fraction update_tau from the main network.
        """

        if self.hyperparams.target_sync_mode == "soft":
            for target_var, var in zip(self.target_network.parameters(), self.main_network.parameters()):
                    target_var.data.copy_((1. - self.hyperparams.update_tau) * target_var.data + (self.hyperparams.update_tau) * var.data)
        else:
            self.target_network.load_state_dict(self.main_network.state_dict())

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

    def get_actions_state(self, obs: TimeStep) -> Dict[str, int]:
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