from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from pysc2.agents import base_agent
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, features, units
from pysc2.lib.features import PlayerRelative
from pysc2.lib.named_array import NamedNumpyArray

from ..actions import AllActions
from ..constants import SC2Costs
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
        AllActions.HARVEST_MINERALS: lambda scv, mineral_unit_tag: actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", scv.tag, mineral_unit_tag),
        AllActions.COLLECT_GAS: lambda scv, refinery_unit_tag: actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", scv.tag, refinery_unit_tag),
        AllActions.BUILD_REFINERY: lambda scv, geyser_position: actions.RAW_FUNCTIONS.Build_refinery_pt("now", scv.tag, geyser_position),
        AllActions.RECRUIT_SCV: lambda command_center:actions.RAW_FUNCTIONS.Train_SCV_quick("now", command_center.tag),
        AllActions.BUILD_SUPPLY_DEPOT: lambda scv, target_position: actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scv.tag, target_position),
        AllActions.BUILD_COMMAND_CENTER: lambda scv, target_position: actions.RAW_FUNCTIONS.Build_CommandCenter_pt("now", scv.tag, target_position),
    }

    @property
    @abstractmethod
    def agent_actions(self) -> List[AllActions]:
        pass

    @abstractmethod
    def select_action(self, obs: TimeStep):
        pass
        # return actions.FUNCTIONS.no_op()

    @abstractmethod
    def get_next_command_center_position(self: TimeStep) -> Position:
        pass

    @abstractmethod
    def get_next_supply_depot_position(self, obs: TimeStep) -> Position:
        pass

    def step(self, obs: TimeStep) -> AllActions:
        obs = self.preprocess_observation(obs)
        action, action_args = self.select_action(obs)

        match action_args:
            case args if isinstance(args, dict):
                return action(**args)
            case _:
                return action()
        # return actions.RAW_FUNCTIONS.no_op()
    
    def preprocess_observation(self, obs: TimeStep) -> TimeStep:
        return obs
    
    def available_actions(self, obs: TimeStep) -> List[AllActions]:
        return [a for a in self.agent_actions if self.can_take(obs, a)]

    def take(self, obs: TimeStep, action: AllActions, *action_args):
        if action not in self._action_to_game:
            action = AllActions.NO_OP
        
        return self._action_to_game[action](*action_args)

    def has_idle_workers(self, player: NamedNumpyArray) -> bool:
        return player.idle_worker_count > 0
    
    def has_workers(self, player: NamedNumpyArray) -> bool:
        return player.food_workers > 0

    def can_take(self, obs: TimeStep, action: AllActions, *action_args) -> bool:
        if action not in self.agent_actions:
            self.logger.warning(f"Tried to validate action {action} that is not available for this agent. Allowed actions: {self.agent_actions}")
            return False
        elif action not in self._action_to_game:
            self.logger.warning(f"Tried to validate action {action} that is not yet implemented in the action to game mapper: {self._action_to_game.keys()}")
            return False

        
        match action, action_args:
            case AllActions.NO_OP, _:
                return True
            case AllActions.HARVEST_MINERALS, (position) if isinstance(position, Position):
                self.logger.info(f"Checking action {action.name} ({action}) with position")
                minerals = [unit.tag for unit in obs.observation.raw_units if Minerals.contains(unit.unit_type) and unit.x == position.x and unit.y == position.y]
                if not any(minerals):
                    self.logger.debug(f"[Action {action.name} ({action}) + position] There are no minerals to harvest at position {position}")
                    return False
                
                if self.has_idle_workers(obs.observation.player):
                    return True
                elif self.has_workers(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action}) + position] Player has no idle SCVs, but has other available SCVs.")
                    return True
                
                self.logger.debug(f"[Action {action.name} ({action}) + position] The target position has minerals, but the player has no SCVs.")
                return False
            case AllActions.HARVEST_MINERALS, _:
                self.logger.info(f"Checking action {action.name} ({action}) with no position")
                minerals = [unit.tag for unit in obs.observation.raw_units if Minerals.contains(unit.unit_type)]
                if not any(minerals):
                    self.logger.debug(f"[Action {action.name} ({action}) without position] There are no minerals on the map")
                    return False
                
                if self.has_idle_workers(obs.observation.player):
                    return True
                elif self.has_workers(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action}) without position] Player has no idle SCVs, but has other available SCVs.")
                    return True
                
                self.logger.debug(f"[Action {action.name} ({action}) without position] There are minerals available, but the player has no SCVs.")
                return False
            case AllActions.BUILD_REFINERY, (position) if isinstance(position, Position):
                geysers = [unit.tag for unit in obs.observation.raw_units if Gas.contains(unit.unit_type) and unit.x == position.x and unit.y == position.y]
                if not any(geysers):
                    self.logger.debug(f"[Action {action.name} ({action}) + position] There are no vespene geysers at position {position} (or they already have a structure)")
                    return False
                
                if not SC2Costs.REFINERY.can_pay(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action}) + position] There are is a vespene geyser at position {position} but the player can't pay the cost ({SC2Costs.REFINERY})")
                    return False

                if self.has_idle_workers(obs.observation.player):
                    return True
                elif self.has_workers(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action}) + position] Player has no idle SCVs, but has other available SCVs.")
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

                if self.has_idle_workers(obs.observation.player):
                    return True
                elif self.has_workers(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action}) without position] Player has no idle SCVs, but has other available SCVs.")
                    return True
                
                self.logger.debug(f"[Action {action.name} ({action}) without position] There are free vespene geysers and the player can pay the cust, but the player has no SCVs.")
                return False
            case AllActions.RECRUIT_SCV, (command_center_tag):
                command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter, unit_tags=command_center_tag)
                if not any(command_centers):
                    self.logger.debug(f"[Action {action.name} ({action})] The player has no command centers")
                    return False
                if command_centers[0].order_length >= 5:
                    self.logger.debug(f"[Action {action.name} ({action})] The command center has the build queue full")
                    return False
                if not SC2Costs.SCV.can_pay(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action})] The player can't pay the cost of an SCV ({SC2Costs.SCV})")
                    return False
                return True
            case AllActions.RECRUIT_SCV, _:
                command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
                if not any(command_centers):
                    self.logger.debug(f"[Action {action.name} ({action})] The player has no command centers")
                    return False
                for command_center in command_centers:
                    if self.can_take(obs, action, command_center.tag):
                        return True
            case AllActions.BUILD_SUPPLY_DEPOT, _:
                target_position = self.get_next_supply_depot_position(obs)
                if target_position is None:
                    self.logger.debug(f"[Action {action.name} ({action})] There are no free positions to build a supply depot")
                    return False
                if not self.has_idle_workers(obs.observation.player):
                    if not self.has_workers(obs.observation.player):
                        self.logger.debug(f"[Action {action.name} ({action})] Player has no SCVs.")
                        return False
                    self.logger.debug(f"[Action {action.name} ({action})] Player has no idle SCVs, but has other available SCVs.")
                if not SC2Costs.SUPPLY_DEPOT.can_pay(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action})] The player can't pay the cost of a Supply Depot ({SC2Costs.SUPPLY_DEPOT})")
                    return False
                
                return True
            case AllActions.BUILD_COMMAND_CENTER, _:
                target_position = self.get_next_command_center_position(obs)
                if target_position is None:
                    self.logger.debug(f"[Action {action.name} ({action})] There are no free positions to build a command center")
                    return False
                if not self.has_idle_workers(obs.observation.player):
                    if not self.has_workers(obs.observation.player):
                        self.logger.debug(f"[Action {action.name} ({action})] Player has no SCVs.")
                        return False
                    self.logger.debug(f"[Action {action.name} ({action})] Player has no idle SCVs, but has other available SCVs.")
                if not SC2Costs.COMMAND_CENTER.can_pay(obs.observation.player):
                    self.logger.debug(f"[Action {action.name} ({action})] The player can't pay the cost of a Command Center ({SC2Costs.COMMAND_CENTER})")
                    return False
                
                return True
            case _:
                self.logger.warning(f"Action {action.name} ({action}) is not implemented yet")
                return False

    def get_self_units(self, obs: TimeStep, unit_types: int | List[int] = None, unit_tags: int | List[int] = None) -> List[features.FeatureUnit]:
        """Get a list of the player's own units.

        Args:
            obs (TimeStep): Observation from the environment
            unit_type (int | List[int], optional): Type of unit(s) to get. If provided, only units of this type(s) will be
                                       returned, otherwise all units are returned.

        Returns:
            List[features.FeatureUnit]: _description_
        """
        units = filter(lambda u: u.alliance == PlayerRelative.SELF, obs.observation.raw_units)

        if unit_types is not None:
            unit_types = [unit_types] if isinstance(unit_types, int) else unit_types
            units = filter(lambda u: u.unit_type in unit_types, units)
        
        if unit_tags is not None:
            unit_tags = [unit_tags] if isinstance(unit_tags, int) else unit_tags
            units = filter(lambda u: u.tag in unit_tags, units)
        
        return list(units)

    def get_idle_scvs(self, obs: TimeStep) -> List[features.FeatureUnit]:
        """Gets all idle SCVs.

        Args:
            obs (TimeStep): Observation from the environment

        Returns:
            List[features.FeatureUnit]: List of idle SCVs
        """
        self_scvs = self.get_self_units(obs, units.Terran.SCV)
        idle_scvs = filter(self_scvs, self.is_idle)

        return idle_scvs

    def get_harvester_scvs(self, obs: TimeStep) -> List[features.FeatureUnit]:
        """Get a list of all SCVs that are currently harvesting.

        Args:
            obs (TimeStep): Observation from the environment.

        Returns:
            List[features.FeatureUnit]: List of SCVs that are harvesting.
        """
        all_scvs = self.get_self_units(obs, units.Terran.SCV)
        return filter(lambda scv: scv.order_id_0 in self.HARVEST_ACTIONS, all_scvs)

    def is_idle(self, unit: features.FeatureUnit) -> bool:
        """Check whether a unit is idle (meaning it has no orders in the queue)"""
        return unit.order_length == 0

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
