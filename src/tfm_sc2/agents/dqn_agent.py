from collections import namedtuple
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, units

from ..actions import AllActions
from ..constants import Constants, SC2Costs
from ..networks.dqn_network import DQNNetwork
from ..networks.experience_replay_buffer import ExperienceReplayBuffer
from ..types import Gas, Minerals
from .base_agent import BaseAgent

DQNAgentParams = namedtuple('DQNAgentParams',
                            field_names=["epsilon", "epsilon_decay", "min_epsilon", "batch_size", "gamma", "loss", "main_network_update_frequency", "target_network_sync_frequency", "target_sync_mode", "update_tau"],
                            defaults=(0.1, 0.99, 0.01, 32, 0.99, None, 1, 50, "soft", 0.001))

State = namedtuple('State',
                            field_names=[
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
    def __init__(self, main_network: DQNNetwork, buffer: ExperienceReplayBuffer,
                 hyperparams: DQNAgentParams,
                 target_network: DQNNetwork = None,
                 train: bool = True,
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
        self._train = train
        self._exploit = not train

        self.initialize()
#         torch.nn.MSELoss()
        self.loss = self.hyperparams.loss or torch.nn.HuberLoss()
        # Placeholders
        self._action_to_idx = None
        self._idx_to_action = None
        self._num_actions = None

        self.__current_state = None
        self.__prev_state = None
        self.__prev_reward = None
        self.__prev_action = None
        self.__prev_action_args = None

        self.__flags = dict(
            burnin_started=False,
            train_started=False,
            exploit_started=False,
            main_net_updated=False,
            target_net_updated=False,
        )

    def initialize(self):
        # This should be exactly the same as self.steps (implemented by pysc2 base agent)
        self.__step_count = 0
        self.__episode_count = 0
        # Loss on each update
        self.__update_losses = []
        # Rewards at the end of each episode
        self.__episode_rewards = []
        # Eps value at the end of each episode
        self.__episode_eps = []
        # Mean rewards at the end of each episode
        self.__mean_rewards = []
        # Rolling average of the rewards of the last 10 episodes
        self.__mean_rewards_10 = []
        # Loss on each episode
        self.__episode_losses = []
        # Steps performed on each episode
        self.__episode_steps = []
        # Average number of steps per episode
        self.__mean_steps = []
        # Highest reward for any episode
        self.__max_episode_rewards = None
        self.__current_episode_reward = 0
        self.__current_episode_steps = 0
        self.__current_episode_losses = []

    def reset(self, **kwargs):
        """Initialize the agent."""
        super().reset(**kwargs)
        self.__episode_count += 1
        # Last observation
        self.__prev_state = None
        self.__prev_reward = None
        self.__prev_action = None
        self.__prev_action_args = None
        self.__current_state = None
        self.__current_episode_reward = 0
        # This should be exactly the same as self.steps (implemented by pysc2 base agent)
        self.__current_episode_steps = 0
        self.__current_episode_losses = []

    @property
    def is_training(self):
        return self._train and (self.buffer.burn_in_capacity >= 1)

    def _convert_obs_to_state(self, obs: TimeStep) -> torch.Tensor:
        building_state = self._get_buildings_state(obs)
        worker_state = self._get_workers_state(obs)
        army_state = self._get_army_state(obs)
        resources_state = self._get_resources_state(obs)
        scores_state = self._get_scores_state(obs)
        neutral_units_state = self._get_neutral_units_state(obs)
        enemy_state = self._get_enemy_state(obs)
        # Enemy

        return torch.Tensor(State(
			**building_state,
			**worker_state,
			**army_state,
			**resources_state,
            **scores_state,
            **neutral_units_state,
            **enemy_state
        )).to(device=self.device)

    def _actions_to_network(self, actions: List[AllActions]) -> List[np.int8]:
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

        return torch.Tensor(ohe_actions).to(device=self.device)


    def select_action(self, obs: TimeStep) -> Tuple[AllActions, Dict[str, Any]]:
        available_actions = self.available_actions(obs)

        self.logger.debug(f"Available actions: {available_actions}")
        # One-hot encoded version of available actions
        valid_actions = self._actions_to_network(available_actions)
        if self._train and (self.buffer.burn_in_capacity < 1):
            if not self.__flags["burnin_started"]:
                self.logger.info(f"Starting burn-in")
                self.__flags["burnin_started"] = True
            self.logger.info(f"Burn in capacity: {100 * self.buffer.burn_in_capacity:.2f}%")
            raw_action = self.main_network.get_random_action(valid_actions=valid_actions)
        elif self.is_training:
            if not self.__flags["train_started"]:
                self.logger.info(f"Starting training")
                self.__flags["train_started"] = True
            raw_action = self.main_network.get_action(self.__current_state, epsilon=self.hyperparams.epsilon, valid_actions=valid_actions)
        else:
            if not self.__flags["exploit_started"]:
                self.logger.info(f"Starting exploit")
                self.__flags["exploit_started"] = True
            raw_action = self.main_network.get_greedy_action(self.__current_state, valid_actions=valid_actions)

        # Convert the "raw" action to a the right type of action
        action = self._idx_to_action[raw_action]

        action_args = self._get_action_args(obs=obs, action=action)

        return action, action_args

    def pre_step(self, obs: TimeStep):
        self.__current_state = self._convert_obs_to_state(obs)
        reward = obs.reward
        self.__current_episode_reward += reward
        self.__current_episode_steps += 1
        # self.__current_episode_losses = []
        # self.__episode_num_steps += 1
        self.__step_count += 1

        if obs.first():
            self._idx_to_action = { idx: action for idx, action in enumerate(self.agent_actions) }
            self._action_to_idx = { action: idx for idx, action in enumerate(self.agent_actions) }
            self._num_actions = len(self.agent_actions)
        else:
            # do updates
            done = obs.last()
            self.buffer.append(self.__prev_state, self.__prev_action, self.__prev_action_args, reward, done, self.__current_state)

            if self.is_training:
                updated = False
                if (self.__current_episode_steps % self.hyperparams.main_network_update_frequency) == 0:
                    if not self.__flags["main_net_updated"]:
                        self.logger.info(f"First main network update")
                        self.__flags["main_net_updated"] = True
                    self.__current_episode_losses = self.update_main_network(self.__current_episode_losses)
                    updated = True
                if (self.__current_episode_steps % self.hyperparams.target_network_sync_frequency) == 0:
                    if not self.__flags["target_net_updated"]:
                        self.logger.info(f"First target network update")
                        self.__flags["target_net_updated"] = True
                    self.synchronize_target_network()
                # HERE

                if done:
                    if not updated:
                        # If we finished but didn't update, perform one last update
                        self.__current_episode_losses = self.update_main_network(self.__current_episode_losses)
                    # Add the episode rewards to the training rewards
                    self.__episode_rewards.append(self.__current_episode_reward)
                    # Register the epsilon used in the last episode played
                    self.__episode_eps.append(self.hyperparams.epsilon)
                    # We'll use the average loss as the episode loss
                    if any(self.__current_episode_losses):
                        episode_loss = sum(self.__current_episode_losses) / len(self.__current_episode_losses)
                    else:
                        self.logger.error("No losses at the end of episode - meaning no updates to the main network happened")
                        episode_loss = np.inf
                    self.__episode_losses.append(episode_loss)
                    self.__episode_steps.append(self.__current_episode_steps)
                    # Get the average reward of all episodes.
                    mean_rewards = np.mean(self.training_rewards)
                    # Also keep the rolling average of the last 10 episodes
                    mean_rewards_10 = np.mean(self.training_rewards[-10:])
                    self.__mean_rewards.append(mean_rewards)
                    self.__mean_rewards_10.append(mean_rewards_10)

                    mean_steps = np.mean(self.episode_steps)
                    self.__mean_steps.append(mean_steps)

                    # Check if we have a new max score
                    if self.__max_episode_rewards is None or (self.__current_episode_reward > self.__max_episode_rewards):
                        self.__max_episode_rewards = self.__current_episode_reward

                    print(
                        f"\rEpisode {self.__episode_count} :: "
                        + f"Mean Rewards ({self.__episode_count} ep) {mean_rewards:.2f} :: "
                        + f"Mean Rewards (10ep) {mean_rewards_10:.2f} :: "
                        + f"Epsilon {self.hyperparams.epsilon:.4f} :: "
                        + f"Max reward {self.__max_episode_rewards:.2f} :: "
                        + f"Episode steps: {self.__current_episode_steps} :: "
                        + f"Total steps: {self.__step_count}\t\t\t\t", end="")

    def post_step(self, obs: TimeStep, action: AllActions, action_args: Dict[str, Any]):
        self.__prev_state = self.__current_state
        self.__prev_action = self._action_to_idx[action]
        self.__prev_action_args = action_args

    def _calculate_loss(self, batch: Iterable[Tuple]) -> torch.Tensor:
        """Calculate the loss for a batch.

        Args:
            batch (Iterable[Tuple]): Batch to calculate the loss on.

        Returns:
            torch.Tensor: The calculated loss between the calculated and the predicted values.
        """

        # Separem les variables de l'experiència i les convertim a tensors
        states, actions, action_args, rewards, dones, next_states = [i for i in batch]
        states = torch.stack(states).to(device=self.device)
        next_states = torch.stack(next_states).to(device=self.device)
        rewards_vals = torch.FloatTensor(rewards).to(device=self.device)
        actions_vals = torch.LongTensor(np.array(actions)).to(device=self.device).reshape(-1,1)
        dones_t = torch.BoolTensor(dones).to(device=self.device)

        # Obtenim els valors de Q de la xarxa principal
        qvals = torch.gather(self.main_network.get_qvals(states), 1, actions_vals)
        # Obtenim els valors de Q de la xarxa objectiu
        # El paràmetre detach() evita que aquests valors actualitzin la xarxa objectiu
        qvals_next = torch.max(self.target_network.get_qvals(next_states), dim=-1)[0].detach()
        qvals_next[dones_t] = 0 # 0 en estats terminals

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

        self.__update_losses.append(loss)
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
        num_harvesters = len([w for w in workers if w.order_id_0 in self.HARVEST_ACTIONS])
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
