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
from ..networks.dqn_network import DQNNetwork
from ..networks.experience_replay_buffer import ExperienceReplayBuffer
from ..types import Gas, Minerals
from .base_agent import BaseAgent
from ..constants import Constants


DQNAgentParams = namedtuple('DQNAgentParams',
                            field_names=["epsilon", "epsilon_decay", "min_epsilon", "batch_size", "gamma", "loss", "main_network_update_frequency", "target_network_sync_frequency", "target_sync_mode", "update_tau"],
                            defaults=(0.1, 0.99, 0.01, 32, 0.99, None, 1, 50, "soft", 0.001))

State = namedtuple('State',
                            field_names=["epsilon", "epsilon_decay", "min_epsilon", "batch_size", "gamma", "loss", "main_network_update_frequency", "target_network_sync_frequency", "target_sync_mode", "update_tau"])

# class DQNState:
#     def from_obs(self, obs: TimeStep) -> Dict:
#         return {
# 			# Command Centers
# 			"num_command_centers": len(command_centers),
# 			"num_completed_command_centers": len(completed_command_centers),
# 			"command_center_0_command_length": queued_scv_1,
#             "command_center_1_command_length": queued_scv_1,
#             "command_center_2_command_length": queued_scv_1,
#             "command_center_3_command_length": queued_scv_1,
#             "command_center_0_num_workers": queued_scv_1,
#             "command_center_1_num_workers": queued_scv_1,
#             "command_center_2_num_workers": queued_scv_1,
#             "command_center_3_num_workers": queued_scv_1,
# 			# SCVs
# 			"num_workers": len(scvs),
#             "num_workers_in_queue": 0,
# 			"num_idle_workers": len(idle_scvs),
#             "num_mineral_harvesters": len(mineral_harvesters),
#             "num_gas_harvesters": len(gas_harvesters),
# 			# TODO more stats on N workers (e.g. distance to command centers, distance to minerals, to geysers...)
# 			# Refineries
# 			"num_refineries": len(refineries),
# 			"num_completed_refineries": len(completed_refineries),
# 			# Supply Depots
# 			"num_supply_depots": len(supply_depots),
# 			"num_completed_supply_depots": len(completed_supply_depots),
# 			# Barracks
# 			"num_barracks": len(barrackses),
# 			"num_completed_barracks": len(completed_barrackses),
# 			# Marines
# 			"num_marines": len(marines),
# 			"num_marines_in_queue": queued_marines,
# 			# Resources
# 			"free_supply": free_supply,
#             "minerals": minerals,
#             "gas": gas,
#             # Scores
#             # Cumulative scores
#             "score_cumulative": obs.observation.score_cumlative._index_names,
#             "score_cumulative_score": obs.observation.score_cumulative.score,
#             "score_cumulative_idle_production_time": obs.observation.score_cumulative.idle_production_time,
#             "score_cumulative_idle_worker_time": obs.observation.score_cumulative.idle_worker_time,
#             "score_cumulative_total_value_units": obs.observation.score_cumulative.total_value_units,
#             "score_cumulative_total_value_structures": obs.observation.score_cumulative.total_value_structures,
#             "score_cumulative_killed_value_units": obs.observation.score_cumulative.killed_value_units,
#             "score_cumulative_killed_value_structures": obs.observation.score_cumulative.killed_value_structures,
#             "score_cumulative_collected_minerals": obs.observation.score_cumulative.collected_minerals,
#             "score_cumulative_collected_vespene": obs.observation.score_cumulative.collected_vespene,
#             "score_cumulative_collection_rate_minerals": obs.observation.score_cumulative.collection_rate_minerals,
#             "score_cumulative_collection_rate_vespene": obs.observation.score_cumulative.collection_rate_vespene,
#             "score_cumulative_spent_minerals": obs.observation.score_cumulative.spent_minerals,
#             "score_cumulative_spent_vespene": obs.observation.score_cumulative.spent_vespene,
#             # Supply (food) scores
#             "score_food_used_none": obs.observation.score_by_category.food_used.none,
#             "score_food_used_army": obs.observation.score_by_category.food_used.army,
#             "score_food_used_economy": obs.observation.score_by_category.food_used.economy,
#             "score_food_used_technology": obs.observation.score_by_category.food_used.technology,
#             "score_food_used_upgrade": obs.observation.score_by_category.food_used.upgrade,
#             "score_by_vital": obs.observation.score_by_vital,
#             # Killed minerals and vespene
#             "score_killed_minerals_none": obs.observation.score_by_category.killed_minerals.none,
#             "score_killed_minerals_army": obs.observation.score_by_category.killed_minerals.army,
#             "score_killed_minerals_economy": obs.observation.score_by_category.killed_minerals.economy,
#             "score_killed_minerals_technology": obs.observation.score_by_category.killed_minerals.technology,
#             "score_killed_minerals_upgrade": obs.observation.score_by_category.killed_minerals.upgrade,
#             "score_killed_minerals_none": obs.observation.score_by_category.killed_minerals.none,
#             "score_killed_vespene_army": obs.observation.score_by_category.killed_vespene.army,
#             "score_killed_vespene_economy": obs.observation.score_by_category.killed_vespene.economy,
#             "score_killed_vespene_technology": obs.observation.score_by_category.killed_vespene.technology,
#             "score_killed_vespene_upgrade": obs.observation.score_by_category.killed_vespene.upgrade,
#             "score_killed_vespene_none": obs.observation.score_by_category.killed_vespene.none,
#             # Lost minerals and vespene
#             "score_lost_minerals_none": obs.observation.score_by_category.lost_minerals.none,
#             "score_lost_minerals_army": obs.observation.score_by_category.lost_minerals.army,
#             "score_lost_minerals_economy": obs.observation.score_by_category.lost_minerals.economy,
#             "score_lost_minerals_technology": obs.observation.score_by_category.lost_minerals.technology,
#             "score_lost_minerals_upgrade": obs.observation.score_by_category.lost_minerals.upgrade,
#             "score_lost_minerals_none": obs.observation.score_by_category.lost_minerals.none,
#             "score_lost_vespene_army": obs.observation.score_by_category.lost_vespene.army,
#             "score_lost_vespene_economy": obs.observation.score_by_category.lost_vespene.economy,
#             "score_lost_vespene_technology": obs.observation.score_by_category.lost_vespene.technology,
#             "score_lost_vespene_upgrade": obs.observation.score_by_category.lost_vespene.upgrade,
#             "score_lost_vespene_none": obs.observation.score_by_category.lost_vespene.none,
#             # Friendly fire minerals and vespene
#             "score_friendly_fire_minerals_none": obs.observation.score_by_category.friendly_fire_minerals.none,
#             "score_friendly_fire_minerals_army": obs.observation.score_by_category.friendly_fire_minerals.army,
#             "score_friendly_fire_minerals_economy": obs.observation.score_by_category.friendly_fire_minerals.economy,
#             "score_friendly_fire_minerals_technology": obs.observation.score_by_category.friendly_fire_minerals.technology,
#             "score_friendly_fire_minerals_upgrade": obs.observation.score_by_category.friendly_fire_minerals.upgrade,
#             "score_friendly_fire_minerals_none": obs.observation.score_by_category.friendly_fire_minerals.none,
#             "score_friendly_fire_vespene_army": obs.observation.score_by_category.friendly_fire_vespene.army,
#             "score_friendly_fire_vespene_economy": obs.observation.score_by_category.friendly_fire_vespene.economy,
#             "score_friendly_fire_vespene_technology": obs.observation.score_by_category.friendly_fire_vespene.technology,
#             "score_friendly_fire_vespene_upgrade": obs.observation.score_by_category.friendly_fire_vespene.upgrade,
#             "score_friendly_fire_vespene_none": obs.observation.score_by_category.friendly_fire_vespene.none,
#             # Used minerals and vespene
#             "score_used_minerals_none": obs.observation.score_by_category.used_minerals.none,
#             "score_used_minerals_army": obs.observation.score_by_category.used_minerals.army,
#             "score_used_minerals_economy": obs.observation.score_by_category.used_minerals.economy,
#             "score_used_minerals_technology": obs.observation.score_by_category.used_minerals.technology,
#             "score_used_minerals_upgrade": obs.observation.score_by_category.used_minerals.upgrade,
#             "score_used_minerals_none": obs.observation.score_by_category.used_minerals.none,
#             "score_used_vespene_army": obs.observation.score_by_category.usede_vespene.army,
#             "score_used_vespene_economy": obs.observation.score_by_category.usede_vespene.economy,
#             "score_used_vespene_technology": obs.observation.score_by_category.usede_vespene.technology,
#             "score_used_vespene_upgrade": obs.observation.score_by_category.usede_vespene.upgrade,
#             "score_used_vespene_none": obs.observation.score_by_category.usede_vespene.none,
#             # Total used minerals and vespene
#             "score_total_used_minerals_none": obs.observation.score_by_category.total_used_minerals.none,
#             "score_total_used_minerals_army": obs.observation.score_by_category.total_used_minerals.army,
#             "score_total_used_minerals_economy": obs.observation.score_by_category.total_used_minerals.economy,
#             "score_total_used_minerals_technology": obs.observation.score_by_category.total_used_minerals.technology,
#             "score_total_used_minerals_upgrade": obs.observation.score_by_category.total_used_minerals.upgrade,
#             "score_total_used_minerals_none": obs.observation.score_by_category.total_used_minerals.none,
#             "score_total_used_vespene_army": obs.observation.score_by_category.total_usede_vespene.army,
#             "score_total_used_vespene_economy": obs.observation.score_by_category.total_usede_vespene.economy,
#             "score_total_used_vespene_technology": obs.observation.score_by_category.total_usede_vespene.technology,
#             "score_total_used_vespene_upgrade": obs.observation.score_by_category.total_usede_vespene.upgrade,
#             "score_total_used_vespene_none": obs.observation.score_by_category.total_usede_vespene.none,

#             # Score by vital
#             "score_by_vital_total_damage_dealt": obs.observation.score_by_vital.total_damage_dealt.life,
#             "score_by_vital_total_damage_dealt": obs.observation.score_by_vital.total_damage_dealt.shields,
#             "score_by_vital_total_damage_dealt": obs.observation.score_by_vital.total_damage_dealt.energy,
#             "score_by_vital_total_damage_taken": obs.observation.score_by_vital.total_damage_taken.life,
#             "score_by_vital_total_damage_taken": obs.observation.score_by_vital.total_damage_taken.shields,
#             "score_by_vital_total_damage_taken": obs.observation.score_by_vital.total_damage_taken.energy,
#             "score_by_vital_total_healed": obs.observation.score_by_vital.total_healed.life,
#             "score_by_vital_total_healed": obs.observation.score_by_vital.total_healed.shields,
#             "score_by_vital_total_healed": obs.observation.score_by_vital.total_healed.energy,

# 			# Enemy
#             # Command Centers
# 			"enemy_num_command_centers": len(command_centers),
# 			"enemy_num_completed_command_centers": len(completed_command_centers),
# 			# SCVs
# 			"enemy_num_workers": len(scvs),
#             # Refineries
# 			"enemy_num_refineries": len(refineries),
# 			# Supply Depots
# 			"enemy_num_supply_depots": len(supply_depots),
# 			# Barracks
# 			"enemy_num_barracks": len(barrackses),
# 			# Marines
# 			"enemy_num_army_units": len(marines),
#         }


class DQNAgent(BaseAgent):
    def __init__(self, main_network: DQNNetwork, buffer: ExperienceReplayBuffer,
                 hyperparams: DQNAgentParams,
                 target_network: DQNNetwork = None
                 ):
        """Deep Q-Network agent.

        Args:
            main_network (nn.Module): Main network
            buffer (ExperienceReplayBuffer): Memory buffer
            hyperparams (DQNAgentParams): Agent hyper parameters.
            target_network (nn.Module, optional): Target network. If not provided, then the main network will be cloned.
        """

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.main_network = main_network
        self.target_network = target_network or deepcopy(main_network)
        self.buffer = buffer
        self.hyperparams = hyperparams
        self.initial_epsilon = hyperparams.epsilon
        self.initialize()
#         torch.nn.MSELoss()
        self.loss = self.hyperparams.loss or torch.nn.HuberLoss()
        # Placeholders
        self.state = None

    def initialize(self):
        # This should be exactly the same as self.steps (implemented by pysc2 base agent)
        self.step_count = 0
        # Loss on each update
        self.update_loss = []
        # Rewards at the end of each episode
        self.training_rewards = []
        # Eps value at the end of each episode
        self.episode_eps = []
        # Mean rewards at the end of each episode
        self.mean_rewards = []
        # Rolling average of the rewards of the last 10 episodes
        self.mean_rewards_10 = []
        # Loss on each episode
        self.episode_losses = []
        # Steps performed on each episode
        self.episode_steps = []
        # Average number of steps per episode
        self.mean_steps = []
        # Highest reward for any episode
        self.max_episode_rewards = None

    def reset(self, **kwargs):
        """Initialize the agent."""
        super().reset(**kwargs)

        # This should be exactly the same as self.steps (implemented by pysc2 base agent)
        self.step_count = 0
        # Last observation
        self.state = None

    def _convert_obs_to_state(self, obs: TimeStep) -> torch.Tensor:
        def _num_complete(buildings):
            return len(list(filter(self.is_complete, buildings)))
        # info about command centers
        command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
        num_command_centers = len(command_centers)
        num_completed_command_centers = _num_complete(command_centers)
        cc_order_lengths = [0, 0, 0, 0]
        cc_num_workers   = [0, 0, 0, 0]

        command_centers_state = dict(
            num_command_centers=num_command_centers,
            num_completed_command_centers=num_completed_command_centers
        )
        for idx, cc in command_centers:
            if idx > 3:
                self.logger.warning(f"Observation space is only ready to consider queues and num workers for 4 command centers, but there are {num_command_centers}.")
                break
            command_centers_state[f"command_center_{idx}_order_length"] = cc.order_length
            command_centers_state[f"command_center_{idx}_num_workers"] = cc.assigned_harvesters

        # Workers
        workers = self.get_workers(obs)
        num_harvesters = [w for w in workers if w.order_id_0 in self.HARVEST_ACTIONS]
        refineries = self.get_self_units(obs, unit_types=units.Terran.Refinery)
        num_gas_harvesters = sum(map(lambda r: r.assigned_harvesters, refineries))
        num_mineral_harvesters = num_harvesters - num_gas_harvesters

        workers_state = dict(
            num_workers=len(workers),
			num_idle_workers=len([w for w in workers if self.is_idle(w)]),
            num_mineral_harvesters=num_mineral_harvesters,
            num_gas_harvesters=num_gas_harvesters,
        )

        # Buildings
        supply_depots = self.get_self_units(obs, unit_types=units.Terran.SupplyDepot)
        barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks)
        buildings_state = dict(
			"num_refineries": len(refineries),
			"num_completed_refineries": _num_complete(refineries),
			# Supply Depots
			"num_supply_depots": len(supply_depots),
			"num_completed_supply_depots": _num_complete(supply_depots),
			# Barracks
			"num_barracks": len(barracks),
			"num_completed_barracks": _num_complete(barracks),
        )

        # Army units
        marines = self.get_self_units(obs, unit_types=units.Terran.Marine)
        num_marines_in_queue = sum(map(lambda b: b.order_length, barracks))
        army_state = dict(
            num_marines=len(marines),
            num_marines_in_queue=num_marines_in_queue
        )

        # Resources
        resources_state = dict(
            free_supply=self.get_free_supply(obs),
            minerals=obs.observation.player.minerals,
            gas=obs.observation.player.vespene,
        )

        # Scores
        scores = {
            "score_cumulative": obs.observation.score_cumlative._index_names,
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
            "score_by_vital": obs.observation.score_by_vital,
            # Killed minerals and vespene
            "score_killed_minerals_none": obs.observation.score_by_category.killed_minerals.none,
            "score_killed_minerals_army": obs.observation.score_by_category.killed_minerals.army,
            "score_killed_minerals_economy": obs.observation.score_by_category.killed_minerals.economy,
            "score_killed_minerals_technology": obs.observation.score_by_category.killed_minerals.technology,
            "score_killed_minerals_upgrade": obs.observation.score_by_category.killed_minerals.upgrade,
            "score_killed_minerals_none": obs.observation.score_by_category.killed_minerals.none,
            "score_killed_vespene_army": obs.observation.score_by_category.killed_vespene.army,
            "score_killed_vespene_economy": obs.observation.score_by_category.killed_vespene.economy,
            "score_killed_vespene_technology": obs.observation.score_by_category.killed_vespene.technology,
            "score_killed_vespene_upgrade": obs.observation.score_by_category.killed_vespene.upgrade,
            "score_killed_vespene_none": obs.observation.score_by_category.killed_vespene.none,
            # Lost minerals and vespene
            "score_lost_minerals_none": obs.observation.score_by_category.lost_minerals.none,
            "score_lost_minerals_army": obs.observation.score_by_category.lost_minerals.army,
            "score_lost_minerals_economy": obs.observation.score_by_category.lost_minerals.economy,
            "score_lost_minerals_technology": obs.observation.score_by_category.lost_minerals.technology,
            "score_lost_minerals_upgrade": obs.observation.score_by_category.lost_minerals.upgrade,
            "score_lost_minerals_none": obs.observation.score_by_category.lost_minerals.none,
            "score_lost_vespene_army": obs.observation.score_by_category.lost_vespene.army,
            "score_lost_vespene_economy": obs.observation.score_by_category.lost_vespene.economy,
            "score_lost_vespene_technology": obs.observation.score_by_category.lost_vespene.technology,
            "score_lost_vespene_upgrade": obs.observation.score_by_category.lost_vespene.upgrade,
            "score_lost_vespene_none": obs.observation.score_by_category.lost_vespene.none,
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
            "score_used_vespene_none": obs.observation.score_by_category.usede_vespene.none,
            "score_used_vespene_army": obs.observation.score_by_category.usede_vespene.army,
            "score_used_vespene_economy": obs.observation.score_by_category.usede_vespene.economy,
            "score_used_vespene_technology": obs.observation.score_by_category.usede_vespene.technology,
            "score_used_vespene_upgrade": obs.observation.score_by_category.usede_vespene.upgrade,
            # Total used minerals and vespene
            "score_total_used_minerals_none": obs.observation.score_by_category.total_used_minerals.none,
            "score_total_used_minerals_army": obs.observation.score_by_category.total_used_minerals.army,
            "score_total_used_minerals_economy": obs.observation.score_by_category.total_used_minerals.economy,
            "score_total_used_minerals_technology": obs.observation.score_by_category.total_used_minerals.technology,
            "score_total_used_minerals_upgrade": obs.observation.score_by_category.total_used_minerals.upgrade,
            "score_total_used_vespene_none": obs.observation.score_by_category.total_usede_vespene.none,
            "score_total_used_vespene_army": obs.observation.score_by_category.total_usede_vespene.army,
            "score_total_used_vespene_economy": obs.observation.score_by_category.total_usede_vespene.economy,
            "score_total_used_vespene_technology": obs.observation.score_by_category.total_usede_vespene.technology,
            "score_total_used_vespene_upgrade": obs.observation.score_by_category.total_usede_vespene.upgrade,

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
        State = namedtuple('State',
                            field_names=[
                                # Command centers
                                "num_command_centers", "num_completed_command_centers",
                                "command_center_0_order_length", "command_center_1_order_length", "command_center_2_order_length", "command_center_3_order_length",
                                "command_center_0_num_workers", "command_center_1_num_workers", "command_center_2_num_workers", "command_center_3_num_workers",
                                # Workers
                                "num_workers", "num_idle_workers", "num_mineral_harvesters", "num_gas_harvesters",
                                # Buildings
                                "num_refineries", "num_completed_refineries", "num_supply_depots", "num_completed_supply_depots", "num_barracks", "num_completed_barracks",
                                # Army
                                "num_marines", "num_marines_in_queue",
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
                                "score_food_used_none", "score_food_used_army", "score_food_used_economy", "score_food_used_technology", "score_food_used_upgrade", "score_by_vital",
                                "score_killed_minerals_none", "score_killed_minerals_army", "score_killed_minerals_economy", "score_killed_minerals_technology", "score_killed_minerals_upgrade", "score_killed_minerals_none",
                                "score_killed_vespene_army", "score_killed_vespene_economy", "score_killed_vespene_technology", "score_killed_vespene_upgrade", "score_killed_vespene_none",
                                "score_lost_minerals_none", "score_lost_minerals_army", "score_lost_minerals_economy", "score_lost_minerals_technology", "score_lost_minerals_upgrade", "score_lost_minerals_none",
                                "score_lost_vespene_army", "score_lost_vespene_economy", "score_lost_vespene_technology", "score_lost_vespene_upgrade", "score_lost_vespene_none",
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
                            ])

        # Neutral units
        minerals = [unit.tag for unit in obs.observation.raw_units if Minerals.contains(unit.unit_type)]
        geysers = [unit.tag for unit in obs.observation.raw_units if Gas.contains(unit.unit_type)]
        neutral_units_state = dict(
            num_minerals=len(minerals),
            num_geysers=len(geysers),
        )

        # Enemy

        enemy_buildings = self.get_enemy_units(obs, unit_types=Constants.BUILDING_UNIT_TYPES)
        enemy_workers = self.get_enemy_units(obs, unit_types=Constants.WORKER_UNIT_TYPES)
        enemy_army = self.get_enemy_units(obs, unit_types=Constants.ARMY_UNIT_TYPES)

        enemy_state = dict(
            num_buildings=len(enemy_buildings),
            num_workers=len(enemy_workers),
            num_army = len(enemy_army),

        )
        return {
			# Command Centers
			**command_centers_state,
            # Workers
			**workers_state,
			# TODO more stats on N workers (e.g. distance to command centers, distance to minerals, to geysers...)
            # Buildings
			**buildings_state,
			# Army
			**army_state,
			# Resources
			**resources_state,
            # Scores
            **scores,

			# Enemy
            # Command Centers
			"enemy_num_command_centers": len(command_centers),
			"enemy_num_completed_command_centers": len(completed_command_centers),
			# SCVs
			"enemy_num_workers": len(scvs),
            # Refineries
			"enemy_num_refineries": len(refineries),
			# Supply Depots
			"enemy_num_supply_depots": len(supply_depots),
			# Barracks
			"enemy_num_barracks": len(barrackses),
			# Marines
			"enemy_num_army_units": len(marines),
        }

    def select_action(self, obs: TimeStep) -> Tuple[AllActions, Dict[str, Any]]:
        available_actions = self.available_actions(obs)
        action = self.main_network.get_action(obs, self.hyperparams.epsilon, available_actions)

        match action:
            case AllActions.NO_OP:
                action_args = None
            case AllActions.HARVEST_MINERALS:
                minerals = [unit for unit in obs.observation.raw_units if Minerals.contains(unit.unit_type)]
                command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
                idle_workers = self.get_idle_workers(obs)

                closest_worker, closest_mineral = self.select_closest_worker(obs, idle_workers, command_centers, minerals)
                action_args = dict(source_unit_tag=closest_worker.tag, target_unit_tag=closest_mineral.tag)
            case AllActions.BUILD_REFINERY:
                geysers = [unit for unit in obs.observation.raw_units if Gas.contains(unit.unit_type)]
                command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
                workers = self.get_idle_workers(obs)
                if len(workers) == 0:
                    workers = self.get_harvester_workers(obs)

                closest_worker, closest_geyser = self.select_closest_worker(obs, workers, command_centers, geysers)
                action_args = dict(source_unit_tag=closest_worker.tag, target_unit_tag=closest_geyser.tag)
            case AllActions.COLLECT_GAS:
                refineries = self.get_self_units(obs, unit_types=[units.Terran.Refinery, units.Terran.RefineryRich])
                command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
                idle_workers = self.get_idle_workers(obs)

                closest_worker, closest_refinery = self.select_closest_worker(obs, idle_workers, command_centers, refineries)
                action_args = dict(source_unit_tag=closest_worker.tag, target_unit_tag=closest_refinery.tag)
            case AllActions.RECRUIT_SCV:
                command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
                command_centers = [cc for cc in command_centers if cc.order_length < Constants.COMMAND_CENTER_QUEUE_LENGTH]
                action_args = dict(source_unit_tag=random.choice(command_centers).tag)
            case AllActions.BUILD_SUPPLY_DEPOT:
                position = self.get_next_supply_depot_position(obs)
                workers = self.get_idle_workers(obs)
                if len(workers) == 0:
                    workers = self.get_harvester_workers(obs)

                worker, _ = self.get_closest(workers, position)
                action_args = dict(source_unit_tag=worker.tag, target_position=position)
            case AllActions.BUILD_COMMAND_CENTER:
                position = self.get_next_command_center_position(obs)
                workers = self.get_idle_workers(obs)
                if len(workers) == 0:
                    workers = self.get_harvester_workers(obs)

                worker, _ = self.get_closest(workers, position)
                action_args = dict(source_unit_tag=worker.tag, target_position=position)
            case AllActions.BUILD_BARRACKS:
                position = self.get_next_barracks_position(obs)
                workers = self.get_idle_workers(obs)
                if len(workers) == 0:
                    workers = self.get_harvester_workers(obs)

                worker, _ = self.get_closest(workers, position)
                action_args = dict(source_unit_tag=worker.tag, target_position=position)
            case AllActions.RECRUIT_MARINE:
                barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks)
                barracks = [cc for cc in barracks if cc.order_length < Constants.BARRACKS_QUEUE_LENGTH]
                action_args = dict(source_unit_tag=random.choice(barracks).tag)
            case AllActions.ATTACK_WITH_SINGLE_UNIT:
                idle_marines = self.get_idle_marines(obs)
                enemies = self.get_enemy_units(obs)
                marine_tag = random.choice(idle_marines).tag
                enemy_tag = random.choice(enemies).tag
                action_args = dict(source_unit_tag=marine_tag, target_unit_tag=enemy_tag)
            case AllActions.ATTACK_WITH_SQUAD_5:
                idle_marines = [m.tag for m in self.get_idle_marines(obs)]
                enemies = self.get_enemy_units(obs)
                enemy_tag = random.choice(enemies).tag
                marine_tags = random.sample(idle_marines, k=5)
                action_args = dict(source_unit_tags=marine_tags, target_unit_tag=enemy_tag)
            case AllActions.ATTACK_WITH_SQUAD_10:
                idle_marines = [m.tag for m in self.get_idle_marines(obs)]
                enemies = self.get_enemy_units(obs)
                enemy_tag = random.choice(enemies).tag
                marine_tags = random.sample(idle_marines, k=10)
                action_args = dict(source_unit_tags=marine_tags, target_unit_tag=enemy_tag)
            case AllActions.ATTACK_WITH_SQUAD_15:
                idle_marines = [m.tag for m in self.get_idle_marines(obs)]
                enemies = self.get_enemy_units(obs)
                enemy_tag = random.choice(enemies).tag
                marine_tags = random.sample(idle_marines, k=15)
                action_args = dict(source_unit_tags=marine_tags, target_unit_tag=enemy_tag)
            case AllActions.ATTACK_WITH_FULL_ARMY:
                idle_marines = [m.tag for m in self.get_idle_marines(obs)]
                enemies = self.get_enemy_units(obs)
                enemy_tag = random.choice(enemies).tag
                action_args = dict(source_unit_tags=idle_marines, target_unit_tag=enemy_tag)
            case _:
                raise RuntimeError(f"Missing logic to select action args for action {action}")

        return action, action_args




    def take_action(self, action: int) -> Tuple[bool, float]:
        """Take an action in the environment.

        If the episode is finished after taking the action, the environment is reset.

        Args:
            action (int): Action to take

        Returns:
            Tuple[bool, float]: A bool indicating if the episode is finished, and a float with the reward of the step.
        """

        new_state, reward, done, truncated, _ = self.env.step(action)
        done = done or truncated

        if not done:
            self.buffer.append(self.state, action, reward, done, new_state)
            self.state = new_state

        if done:
            self.state = self.env.reset()[0]

        return done, reward

    def take_step(self, eps: float) -> Tuple[bool, float]:
        """Perform a step in the environment.

        The action will be selected from the main network, and will have a probability
        "eps" of taking a random action.

        Args:
            eps (float): Probability of taking a random action

        Returns:
            Tuple[bool, float]: A bool indicating if the episode is finished, and a float with the reward of the step.
        """
        action = self.main_network.get_action(self.state, eps)
        done, reward = self.take_action(action)

        return done, reward

    def play_episode(self, episode_number: int, mode="train") -> Tuple[int, float, float]:
        """Play a full episode.

        If mode is "train", extra metrics are captured, to be used later.

        Args:
            episode_number (int): Episode number, only used for informative purposes.
            mode (str): Set to "train" during training

        Returns:
            Tuple[int, float, float]: A tuple with (number of steps, episode reward, mean rewards of last 100 episodes)
        """

        self.state = self.env.reset()[0]
        num_steps = 0
        episode_reward = 0
        done = False
        is_training = mode == "train"
        mean_rewards = None
        episode_losses = []

        # In not training, use an epsilon of 0
        eps = self.hyperparams.epsilon if is_training else 0
        while not done:
            done, reward = self.take_step(eps)
            self.step_count += 1

            num_steps += 1
            episode_reward += reward

            if is_training:
                episode_losses = self.update_main_network(episode_losses)
                self.synchronize_target_network()

        if is_training:
            # Add the episode rewards to the training rewards
            self.training_rewards.append(episode_reward)
            # Register the epsilon used in the last episode played
            self.episode_eps.append(self.hyperparams.epsilon)
            # We'll use the average loss as the episode loss
            if any(episode_losses):
                episode_loss = sum(episode_losses) / len(episode_losses)
            else:
                episode_loss = 100
            self.episode_losses.append(episode_loss)
            self.episode_steps.append(num_steps)
            """
            Get the average reward of the last episodes. Here we keep track
            of the rolling average of the las N episodes, where N is the minimum number
            of episodes we need to solve the environment.
            """
            mean_rewards = np.mean(self.training_rewards[-self.solve_num_episodes:])
            # Also keep the rolling average of the last 10 episodes
            mean_rewards_10 = np.mean(self.training_rewards[-10:])
            self.mean_rewards.append(mean_rewards)
            self.mean_rewards_10.append(mean_rewards_10)

            mean_steps = np.mean(self.episode_steps[-self.solve_num_episodes:])
            self.mean_steps.append(mean_steps)

            # Check if we have a new max score
            if self.max_episode_rewards is None or (episode_reward > self.max_episode_rewards):
                self.max_episode_rewards = episode_reward

            print(
                f"\rEpisode {episode_number} :: "
                + f"Mean Rewards ({self.solve_num_episodes} ep) {mean_rewards:.2f} :: "
                + f"Mean Rewards (10ep) {mean_rewards_10:.2f} :: "
                + f"Epsilon {self.hyperparams.epsilon:.4f} :: "
                + f"Maxim {self.max_episode_rewards:.2f} :: "
                + f"Steps: {num_steps}\t\t\t\t", end="")

        self.env.close()

        return num_steps, episode_reward, mean_rewards

    def train(self, max_episodes: int):
        """Train the agent for a certain number of episodes.

        Depending on the settings, the agent might stop training as soon as the environment is solved,
        or it might continue training despite having solved it.

        Args:
            max_episodes (int): Maximum number of episodes to play.
        """

        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity < 1:
            action = self.main_network.get_random_action()
            self.take_action(action)

        episode = 0
        training = True
        print("Training...")
        max_reward = None

        best_main_net = deepcopy(self.main_network)
        best_target_net = deepcopy(self.target_network)
        best_episode = 0
        best_mean_rewards = -10000

        solved = False
        self.hyperparams.epsilon = self.initial_epsilon

        while training:
            # Play an episode
            episode_steps, episode_reward, mean_rewards = self.play_episode(episode_number=episode)
            episode += 1

            # If we reached the maximum number of episodes, finish the training
            if episode >= max_episodes:
                training = False
                print("\nEpisode limit reached.")
                break

            min_episodes_reached = self.solve_num_episodes <  episode
            new_best_mean_reward = mean_rewards > best_mean_rewards
            solve_threshold_reached = mean_rewards >= self.solve_reward_threshold

            # Once the minimum number of episodes is reached, keep track of the best model
            if min_episodes_reached and new_best_mean_reward:
                best_main_net.load_state_dict(self.main_network.state_dict())
                best_target_net.load_state_dict(self.target_network.state_dict())
                best_episode = episode
                best_mean_rewards = mean_rewards

            # Check if the environment has been solved
            if solve_threshold_reached and min_episodes_reached:
                # We only trigger this once, in case we keep training after having solved the environment
                if not solved:
                    print(f'\nEnvironment solved in {episode} episodes!')
                    solved = True

                # On the other hand, if the environment is solved and we should not continue training
                # after solving, we are done training.
                if not self.keep_training_after_solving:
                    training = False
                    break

            self.hyperparams.epsilon = max(self.hyperparams.epsilon * self.hyperparams.epsilon_decay, self.hyperparams.min_epsilon)

        # If keep_training_after_solving is true, we restore the agent to the version
        # that achieved the best mean rewards
        if self.restore_to_best_version:
            print(f"\nFinished training after {episode} episodes, restoring agent to best version at episode {best_episode}")
            print(f"Best agent got mean rewards {best_mean_rewards:.2f} at episode {best_episode}")

            self.main_network.load_state_dict(best_main_net.state_dict())
            self.target_network.load_state_dict(best_target_net.state_dict())
            self.training_rewards = self.training_rewards[:best_episode]
            self.mean_rewards = self.mean_rewards[:best_episode]
            self.mean_rewards_10 = self.mean_rewards_10[:best_episode]
            self.episode_eps = self.episode_eps[:best_episode]
            self.episode_losses = self.episode_losses[:best_episode]
            self.episode_steps = self.episode_steps[:best_episode]

    def calculate_loss(self, batch: Iterable[Tuple]) -> torch.Tensor:
        """Calculate the loss for a batch.

        Args:
            batch (Iterable[Tuple]): Batch to calculate the loss on.

        Returns:
            torch.Tensor: The calculated loss between the calculated and the predicted values.
        """

        # Separem les variables de l'experiència i les convertim a tensors
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_vals = torch.FloatTensor(rewards).to(device=self.device)
        actions_vals = torch.LongTensor(np.array(actions)).to(device=self.device).reshape(-1,1)
        dones_t = torch.ByteTensor(dones).to(device=self.device)

        # Obtenim els valors de Q de la xarxa principal
        qvals = torch.gather(self.main_network.get_qvals(states), 1, actions_vals)
        # Obtenim els valors de Q de la xarxa objectiu
        # El paràmetre detach() evita que aquests valors actualitzin la xarxa objectiu
        qvals_next = torch.max(self.target_network.get_qvals(next_states),
                               dim=-1)[0].detach()
        qvals_next[dones_t] = 0 # 0 en estats terminals

        expected_qvals = self.hyperparams.gamma * qvals_next + rewards_vals

        return self.loss(qvals, expected_qvals.reshape(-1,1))

    def update_main_network(self, episode_losses: List[float] = None, force_update: bool = False) -> List[float]:
        """Update the main network.

        Normally we only perform the update every certain number of steps, defined by
        main_network_update_frequency, but if force_update is set to true, then
        the update will happen, independently of the current step count.

        Args:
            force_update (bool, optional): Force update of the network without checking the step count. Defaults to False.
        Returns:
            List[float]: A list with all the losses in the current episode.
        """
        episode_losses = episode_losses or []
        if (self.steps % self.hyperparams.main_network_update_frequency != 0) and not force_update:
            return episode_losses

        self.main_network.optimizer.zero_grad()  # eliminem qualsevol gradient passat
        batch = self.buffer.sample_batch(batch_size=self.hyperparams.batch_size) # seleccionem un conjunt del buffer

        loss = self.calculate_loss(batch)# calculem la pèrdua
        loss.backward() # calculem la diferència per obtenir els gradients
        self.main_network.optimizer.step() # apliquem els gradients a la xarxa neuronal

        if self.device == 'cuda':
            loss = loss.detach().cpu().numpy()
        else:
            loss = loss.detach().numpy()

        self.update_loss.append(loss)
        episode_losses.append(float(loss))
        return episode_losses

    def synchronize_target_network(self, force_update: bool = False):
        """Synchronize the target network with the main network parameters.

        When the target_sync_mode is set to "soft", a soft update is made, so instead of overwriting
        the target fully, we update it by mixing in the current target parameters and the main network
        parameters. In practice, we keep a fraction (1 - update_tau) from the target network, and add
        to it a fraction update_tau from the main network.

        Args:
            force_update (bool, optional): If true, the target network will be synched, no matter the step count. Defaults to False.
        """
        if (self.steps % self.hyperparams.target_network_sync_frequency != 0) and not force_update:
            return

        if self.hyperparams.target_sync_mode == "soft":
            for target_var, var in zip(self.target_network.parameters(), self.main_network.parameters()):
                    target_var.data.copy_((1. - self.hyperparams.update_tau) * target_var.data + (self.hyperparams.update_tau) * var.data)
        else:
            self.target_network.load_state_dict(self.main_network.state_dict())
