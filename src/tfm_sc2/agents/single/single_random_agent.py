import random
from typing import Any, Dict, List, Tuple

from pysc2.env.environment import TimeStep
from pysc2.lib import actions, units

from ...actions import AllActions
from ...constants import Constants
from ..base_agent import BaseAgent

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS
from ...types import Gas, Minerals


class SingleRandomAgent(BaseAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__agent_actions = list(AllActions)

    @property
    def agent_actions(self) -> List[AllActions]:
        return self.__agent_actions

    def select_action(self, obs: TimeStep) -> Tuple[AllActions, Dict[str, Any]]:
        available_actions = self.available_actions(obs)
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
