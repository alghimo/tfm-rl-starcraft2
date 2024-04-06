import random
from typing import Any, Dict, List, Tuple

import numpy as np
from absl import app
from pysc2.env import sc2_env
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, features, units

from ...actions import AllActions, BaseManagerActions, ResourceManagerActions
from ...types import Position

# from ..utils import enemy_locs, self_locs, xy_locs
from ..base_agent import BaseAgent

# _PLAYER_SELF = features.PlayerRelative.SELF
# _PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
# _PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS
from ...types import Gas, Minerals


class TestAgent(BaseAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__agent_actions = list(set(list(ResourceManagerActions) + list(BaseManagerActions)))

    @property
    def agent_actions(self) -> List[AllActions]:
        return [AllActions.NO_OP, AllActions.HARVEST_MINERALS, AllActions.BUILD_REFINERY, AllActions.COLLECT_GAS, AllActions.RECRUIT_SCV, AllActions.BUILD_SUPPLY_DEPOT, AllActions.BUILD_COMMAND_CENTER]
        return self.__agent_actions

    def select_action(self, obs: TimeStep) -> Tuple[AllActions, Dict[str, Any]]:
        # command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
        # print(f"Command center at position ", (command_centers[0].x, command_centers[0].y))
        # geysers = [unit for unit in obs.observation.raw_units if Gas.contains(unit.unit_type)]
        # minerals = [unit for unit in obs.observation.raw_units if Minerals.contains(unit.unit_type)]
        # for geyser in geysers:
        #     print(f"Geyser at position ", (geyser.x, geyser.y))
        # for mineral in minerals:
        #     print(f"Mineral at position ", (mineral.x, mineral.y))
        # import pdb
        # pdb.set_trace()
        available_actions = self.available_actions(obs)

        if AllActions.BUILD_SUPPLY_DEPOT in available_actions:
            action = AllActions.BUILD_SUPPLY_DEPOT
        else:
            action = random.choice(available_actions)

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
                command_centers = [cc for cc in command_centers if cc.order_length < 5]
                action_args = dict(source_unit_tag=random.choice(command_centers).tag)
            case AllActions.BUILD_SUPPLY_DEPOT:
                position = self.take_next_supply_depot_position(obs)
                workers = self.get_idle_workers(obs)
                if len(workers) == 0:
                    workers = self.get_harvester_workers(obs)

                worker, _ = self.get_closest(workers, position)
                action_args = dict(source_unit_tag=worker.tag, target_position=position)
            case AllActions.BUILD_COMMAND_CENTER:
                position = self.take_next_command_center_position(obs)
                workers = self.get_idle_workers(obs)
                if len(workers) == 0:
                    workers = self.get_harvester_workers(obs)

                worker, _ = self.get_closest(workers, position)
                action_args = dict(source_unit_tag=worker.tag, target_position=position)
            case _:
                raise RuntimeError(f"Missing logic to select action args for action {action}")

        return action, action_args

    # def get_next_command_center_position(self, obs: TimeStep) -> Position:
    #     return None

    # def get_next_supply_depot_position(self, obs: TimeStep) -> Position:
    #     return None
