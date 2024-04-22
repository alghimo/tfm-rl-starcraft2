import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
from pysc2.agents import base_agent
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, features, units
from pysc2.lib.features import PlayerRelative
from pysc2.lib.named_array import NamedNumpyArray

from ..actions import AllActions
from ..constants import Constants, SC2Costs
from ..types import Gas, Minerals, Position
from ..with_logger import WithLogger

# from .agent_utils import AgentUtils


class BaseAgent(WithLogger, ABC, base_agent.BaseAgent):
    HARVEST_ACTIONS = [
        359, # Function.raw_ability(359, "Harvest_Gather_SCV_unit", raw_cmd_unit, 295, 3666),
        362, # Function.raw_ability(362, "Harvest_Return_SCV_quick", raw_cmd, 296, 3667),
    ]

    _action_to_game = {
        AllActions.NO_OP: actions.RAW_FUNCTIONS.no_op,
        AllActions.HARVEST_MINERALS: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", source_unit_tag, target_unit_tag),
        AllActions.COLLECT_GAS: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", source_unit_tag, target_unit_tag),
        AllActions.BUILD_REFINERY: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Build_Refinery_pt("now", source_unit_tag, target_unit_tag),
        AllActions.RECRUIT_SCV: lambda source_unit_tag: actions.RAW_FUNCTIONS.Train_SCV_quick("now", source_unit_tag),
        AllActions.BUILD_SUPPLY_DEPOT: lambda source_unit_tag, target_position: actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", source_unit_tag, target_position),
        AllActions.BUILD_COMMAND_CENTER: lambda source_unit_tag, target_position: actions.RAW_FUNCTIONS.Build_CommandCenter_pt("now", source_unit_tag, target_position),
        AllActions.BUILD_BARRACKS: lambda source_unit_tag, target_position: actions.RAW_FUNCTIONS.Build_Barracks_pt("now", source_unit_tag, target_position),
        AllActions.RECRUIT_MARINE: lambda source_unit_tag: actions.RAW_FUNCTIONS.Train_Marine_quick("now", source_unit_tag),
        AllActions.ATTACK_WITH_SINGLE_UNIT: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tag, target_unit_tag),
        AllActions.ATTACK_WITH_SQUAD_5: lambda source_unit_tags, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tags, target_unit_tag),
        AllActions.ATTACK_WITH_SQUAD_10: lambda source_unit_tags, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tags, target_unit_tag),
        AllActions.ATTACK_WITH_SQUAD_15: lambda source_unit_tags, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tags, target_unit_tag),
        AllActions.ATTACK_WITH_FULL_ARMY: lambda source_unit_tags, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tags, target_unit_tag)
    }

    def __init__(self, map_name: str, map_config: Dict, **kwargs):
        super().__init__(**kwargs)
        self._map_name = map_name
        self._map_config = map_config
        self._supply_depot_positions = None
        self._command_center_positions = None
        self._barrack_positions = None
        self._used_supply_depot_positions = None
        self._used_command_center_positions = None
        self._used_barrack_positions = None

    def _load_agent_attrs(self, agent_attrs: Dict):
        self._map_name = agent_attrs["map_name"]
        self._map_config = agent_attrs["map_config"]
        self._supply_depot_positions = agent_attrs["supply_depot_positions"]
        self._command_center_positions = agent_attrs["command_center_positions"]
        self._barrack_positions = agent_attrs["barrack_positions"]
        self._used_supply_depot_positions = agent_attrs["used_supply_depot_positions"]
        self._used_command_center_positions = agent_attrs["used_command_center_positions"]
        self._used_barrack_positions = agent_attrs["used_barrack_positions"]
        # From SC2's Base agent
        self.reward = agent_attrs["reward"]
        self.episodes = agent_attrs["episodes"]
        self.steps = agent_attrs["steps"]
        self.obs_spec = agent_attrs["obs_spec"]
        self.action_spec = agent_attrs["action_spec"]
        # From WithLogger
        self._log_name = agent_attrs["log_name"]
        if self.logger.name != self._log_name:
            self._logger = logging.getLogger(self._log_name)

    def _get_agent_attrs(self):
        return dict(
            map_name=self._map_name,
            map_config=self._map_config,
            supply_depot_positions=self._supply_depot_positions,
            used_supply_depot_positions=self._used_supply_depot_positions,
            used_command_center_positions=self._used_command_center_positions,
            used_barrack_positions=self._used_barrack_positions,
            command_center_positions=self._command_center_positions,
            barrack_positions=self._barrack_positions,
            # From SC2's Base agent
            reward=self.reward,
            episodes=self.episodes,
            steps=self.steps,
            obs_spec=self.obs_spec,
            action_spec=self.action_spec,
            # From logger
            log_name=self._log_name,
        )

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._supply_depot_positions = None
        self._barrack_positions = None
        self._command_center_positions = None
        self._used_supply_depot_positions = None
        self._used_command_center_positions = None
        self._used_barrack_positions = None

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
                    closest_worker, closest_mineral = self.select_closest_worker(obs, idle_workers, command_centers, minerals)
                    action_args = dict(source_unit_tag=closest_worker.tag, target_unit_tag=closest_mineral.tag)
                case AllActions.BUILD_REFINERY:
                    geysers = [unit for unit in obs.observation.raw_units if Gas.contains(unit.unit_type)]
                    command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
                    workers = self.get_idle_workers(obs)
                    if len(workers) == 0:
                        workers = self.get_harvester_workers(obs)

                    if len(command_centers) == 0:
                        # Only takes into account distance between workers and geysers
                        closest_worker, closest_geyser = self.select_closest_worker_to_resource(obs, workers, geysers)
                    else:
                        # Takes into account the distance from the worker to the command center and from the command center to the geyser
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

            return action_args, True
        except:
            return None, False

    def get_next_command_center_position(self, obs: TimeStep) -> Position:
        next_pos = None
        for idx, candidate_position in enumerate(self._command_center_positions):
            if candidate_position not in self._used_command_center_positions:
                next_pos = candidate_position
                break

        if next_pos is not None:
            # put all positions before the candidate position at the end
            self._command_center_positions = self._command_center_positions[idx + 1:] + self._command_center_positions[:idx+1]

        return next_pos

    def update_command_center_positions(self, obs: TimeStep) -> Position:
        command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
        command_center_positions = [Position(cc.x, cc.y) for cc in command_centers]
        self._used_command_center_positions = command_center_positions
        self._command_center_positions = [pos for pos in self._command_center_positions if pos not in command_center_positions]

    def get_next_supply_depot_position(self, obs: TimeStep) -> Position:
        next_pos = None
        for idx, candidate_position in enumerate(self._supply_depot_positions):
            if candidate_position not in self._used_supply_depot_positions:
                next_pos = candidate_position
                break

        if next_pos is not None:
            # put all positions before the candidate position at the end
            self._supply_depot_positions = self._supply_depot_positions[idx + 1:] + self._supply_depot_positions[:idx+1]

        return next_pos

    def update_supply_depot_positions(self, obs: TimeStep) -> Position:
        supply_depots = self.get_self_units(obs, unit_types=units.Terran.SupplyDepot)
        supply_depots_positions = [Position(sd.x, sd.y) for sd in supply_depots]
        self._used_supply_depot_positions = supply_depots_positions
        self._supply_depot_positions = [pos for pos in self._supply_depot_positions if pos not in supply_depots_positions]

    def get_next_barracks_position(self, obs: TimeStep) -> Position:
        next_pos = None
        for idx, candidate_position in enumerate(self._barrack_positions):
            if candidate_position not in self._used_barrack_positions:
                next_pos = candidate_position
                break

        if next_pos is not None:
            # put all positions before the candidate position at the end
            self._barrack_positions = self._barrack_positions[idx + 1:] + self._barrack_positions[:idx+1]

        return next_pos

    def update_barracks_positions(self, obs: TimeStep) -> Position:
        barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks)
        barrack_positions = [Position(b.x, b.y) for b in barracks]
        self._used_barrack_positions = barrack_positions
        self._barrack_positions = [pos for pos in self._supply_depot_positions if pos not in barrack_positions]

    def pre_step(self, obs: TimeStep):
        pass

    def post_step(self, obs: TimeStep, action: AllActions, action_args: Dict[str, Any]):
        pass

    def step(self, obs: TimeStep) -> AllActions:
        self.pre_step(obs)

        super().step(obs)
        if obs.first():
            self._setup_positions(obs)

        self.update_supply_depot_positions(obs)
        self.update_command_center_positions(obs)
        self.update_barracks_positions(obs)

        obs = self.preprocess_observation(obs)
        action, action_args = self.select_action(obs)

        self.logger.debug(f"[Step {self.steps}] Performing action {action.name} with args: {action_args}")
        game_action = self._action_to_game[action]

        self.post_step(obs, action, action_args)
        if action_args is not None:
            return game_action(**action_args)

        return game_action()

    def preprocess_observation(self, obs: TimeStep) -> TimeStep:
        return obs

    def available_actions(self, obs: TimeStep) -> List[AllActions]:
        return [a for a in self.agent_actions if self.can_take(obs, a)]

    def take(self, obs: TimeStep, action: AllActions, *action_args):
        if action not in self._action_to_game:
            action = AllActions.NO_OP

        return self._action_to_game[action](*action_args)

    def get_idle_marines(self, obs: TimeStep) -> List[features.FeatureUnit]:
        """Gets all idle workers.

        Args:
            obs (TimeStep): Observation from the environment

        Returns:
            List[features.FeatureUnit]: List of idle workers
        """
        self_marines = self.get_self_units(obs, units.Terran.Marine)
        idle_marines = filter(self.is_idle, self_marines)

        return list(self_marines)

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
        if action not in self.agent_actions:
            self.logger.warning(f"Tried to validate action {action} that is not available for this agent. Allowed actions: {self.agent_actions}")
            return False
        elif action not in self._map_config["available_actions"]:
            self.logger.debug(f"Action {action} that is not available for this map.")
            return False
        elif action not in self._action_to_game:
            self.logger.warning(f"Tried to validate action {action} that is not yet implemented in the action to game mapper: {self._action_to_game.keys()}")
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
            case AllActions.COLLECT_GAS, args if _has_target_unit_tag(args):
                target_unit_tag = args["target_unit_tag"]
                self.logger.debug(f"Checking action {action.name} ({action}) with position")
                command_centers = [unit.tag for unit in self.get_self_units(obs, unit_types=units.Terran.CommandCenter)]
                if not any(command_centers):
                    self.logger.debug(f"[Action {action.name} ({action}) + position] The player has no command centers")
                    return False

                refineries = self.get_self_units(obs, unit_types=[units.Terran.Refinery, units.Terran.RefineryRich], unit_tags=target_unit_tag)

                if len(refineries) == 0:
                    self.logger.debug(f"[Action {action.name} ({action}) + position] There are no refineries to harvest at position {position}")
                    return False

                if self.has_idle_workers(obs):
                    return True
                # TODO Add check for workers harvesting minerals

                self.logger.debug(f"[Action {action.name} ({action}) + position] The target position has minerals, but the player has no SCVs.")
                return False
            case AllActions.COLLECT_GAS, _:
                self.logger.debug(f"Checking action {action.name} ({action}) with no position")
                command_centers = [unit.tag for unit in self.get_self_units(obs, unit_types=units.Terran.CommandCenter)]
                if not any(command_centers):
                    self.logger.debug(f"[Action {action.name} ({action}) + position] The player has no command centers")
                    return False

                refineries = self.get_self_units(obs, unit_types=[units.Terran.Refinery, units.Terran.RefineryRich])
                refineries = [unit.tag for unit in refineries]
                if not any(refineries):
                    self.logger.debug(f"[Action {action.name} ({action}) without position] There are no refineries on the map")
                    return False

                if self.has_idle_workers(obs):
                    return True
                # elif self.has_workers(obs):
                #     return True
                # TODO Add check for workers harvesting minerals

                self.logger.debug(f"[Action {action.name} ({action}) without position] There are refineries, but the player has no workers.")
                return False
            # TODO Add this check
            # case AllActions.BUILD_REFINERY, args if _has_source_unit_tag(args) and _has_target_unit_tag(args):
            case AllActions.BUILD_REFINERY, args if _has_target_unit_tag(args):
                target_unit_tag = args["target_unit_tag"]
                geysers = [unit.tag for unit in obs.observation.raw_units if unit.tag == target_unit_tag and Gas.contains(unit.unit_type)]
                if not any(geysers):
                    self.logger.debug(f"[Action {action.name} ({action}) + position] There are no vespene geysers at position {position} (or they already have a structure)")
                    return False

                if not SC2Costs.REFINERY.can_pay(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action}) + position] There are is a vespene geyser at position {position} but the player can't pay the cost ({SC2Costs.REFINERY})")
                    return False

                if self.has_idle_workers(obs):
                    return True
                elif self.has_harvester_workers(obs):
                    self.logger.debug(f"[Action {action.name} ({action}) + position] Player has no idle SCVs, but has other available workers harvesting.")
                    return True

                self.logger.debug(f"[Action {action.name} ({action}) + position] The target position has a vespene geyser and the player can pay the cust, but the player has no SCVs.")
                return False
            case AllActions.BUILD_REFINERY, _:
                geysers = [unit.tag for unit in obs.observation.raw_units if Gas.contains(unit.unit_type)]
                if not any(geysers):
                    self.logger.debug(f"[Action {action.name} ({action}) without position] There are no vespene geysers on the map(or they already have a structure)")
                    return False

                if not SC2Costs.REFINERY.can_pay(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action}) without position] There are are vespene geysers available but the player can't pay the cost ({SC2Costs.REFINERY})")
                    return False

                if self.has_idle_workers(obs):
                    return True
                elif self.has_workers(obs):
                    self.logger.debug(f"[Action {action.name} ({action}) without position] Player has no idle SCVs, but has other available SCVs.")
                    return True

                self.logger.debug(f"[Action {action.name} ({action}) without position] There are free vespene geysers and the player can pay the cust, but the player has no SCVs.")
                return False
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
            case AllActions.ATTACK_WITH_SQUAD_5, _:
                self.logger.debug(f"Checking action {action.name} ({action}) without source or target unit tags")
                idle_marines = self.get_idle_marines(obs)
                if len(idle_marines) < 5:
                    self.logger.debug(f"[Action {action.name} ({action})] The player has less than 5 idle marines")
                    return False
                enemies = self.get_enemy_units(obs)
                if len(enemies) == 0:
                    self.logger.debug(f"[Action {action.name} ({action})] The are no enemies to attack")
                    return False

                return True
            case AllActions.ATTACK_WITH_SQUAD_10, _:
                self.logger.debug(f"Checking action {action.name} ({action}) without source or target unit tags")
                idle_marines = self.get_idle_marines(obs)
                if len(idle_marines) < 10:
                    self.logger.debug(f"[Action {action.name} ({action})] The player has less than 10 idle marines")
                    return False
                enemies = self.get_enemy_units(obs)
                if len(enemies) == 0:
                    self.logger.debug(f"[Action {action.name} ({action})] The are no enemies to attack")
                    return False

                return True
            case AllActions.ATTACK_WITH_SQUAD_15, _:
                self.logger.debug(f"Checking action {action.name} ({action}) without source or target unit tags")
                idle_marines = self.get_idle_marines(obs)
                if len(idle_marines) < 15:
                    self.logger.debug(f"[Action {action.name} ({action})] The player has less than 15 idle marines")
                    return False
                enemies = self.get_enemy_units(obs)
                if len(enemies) == 0:
                    self.logger.debug(f"[Action {action.name} ({action})] The are no enemies to attack")
                    return False

                return True
            case AllActions.ATTACK_WITH_FULL_ARMY, _:
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
            self.logger.error(f"Asking for workers that are idle AND harvesting will always result in an empty list")
            return []

        workers = self.get_self_units(obs, units.Terran.SCV)

        if idle:
            workers = filter(self.is_idle, workers)
        elif harvesting:
            workers = filter(lambda w: w.order_id_0 in self.HARVEST_ACTIONS, workers)

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

    # def select_target_enemy(self, enemies: List[Position], obs: TimeStep, **kwargs):
    #     """Given a list of enemies, selects one of them.

    #     Args:
    #         enemies (List[Position]): List of enemies, usually obtained via self.get_enemy_positions.
    #         obs (TimeStep): Observation, can be used for conext or as support to make the decision.

    #     Returns:
    #         Position: The Position of the selected enemy.
    #     """

    #     # Simply return the first enemy
    #     return enemies[np.argmax(np.array(enemies)[:, 1])]

    # def take_action(self)
