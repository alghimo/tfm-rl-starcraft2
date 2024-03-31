from typing import TYPE_CHECKING, List

import numpy as np
from pysc2.agents import base_agent
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, features, units
from pysc2.lib.features import PlayerRelative

from ..types import Position
from .agent_utils import AgentUtils


class BaseAgent(AgentUtils, base_agent.BaseAgent):
    HARVEST_ACTIONS = [
        359, # Function.raw_ability(359, "Harvest_Gather_SCV_unit", raw_cmd_unit, 295, 3666),
        362, # Function.raw_ability(362, "Harvest_Return_SCV_quick", raw_cmd, 296, 3667),
    ]
    
    def step(self, obs: TimeStep):
        obs = self.preprocess_observation(obs)
        action = self.select_action(obs)

        return action
    
    def preprocess_observation(self, obs: TimeStep):
        return obs
    
    def select_action(self, obs):
        return actions.FUNCTIONS.no_op()

    def select_target_enemy(self, enemies: List[Position], obs: TimeStep, **kwargs):
        """Given a list of enemies, selects one of them.

        Args:
            enemies (List[Position]): List of enemies, usually obtained via self.get_enemy_positions.
            obs (TimeStep): Observation, can be used for conext or as support to make the decision.

        Returns:
            Position: The Position of the selected enemy.
        """

        # Simply return the first enemy
        return enemies[np.argmax(np.array(enemies)[:, 1])]
