from typing import List

from absl import app
import numpy as np
from pysc2.env import sc2_env
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, features

from .agent_actions import ResourceManagerActions
from ..base_agent import BaseAgent

class ResourceManager(BaseAgent):
    HARVEST_ACTIONS = [
        359, # Function.raw_ability(359, "Harvest_Gather_SCV_unit", raw_cmd_unit, 295, 3666),
        362, # Function.raw_ability(362, "Harvest_Return_SCV_quick", raw_cmd, 296, 3667),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._idx_to_action = {idx: a for (idx, a) in enumerate(list(ResourceManagerActions))}
        self._action_to_idx = {a: idx for (idx, a) in enumerate(list(ResourceManagerActions))}
        self._scvs = {}

    def actions(self) -> List[ResourceManagerActions]:
        return list(self._idx_to_action.values())

    def action_to_idx(self, action: ResourceManagerActions) -> int:
        return self._action_to_idx[action]

    def action_to_idx(self, action: int) -> ResourceManagerActions:
        return self._idx_to_action[action]
    
    def preprocess_observation(self, obs: TimeStep):
        return super().preprocess_observation(obs)
    
    def select_action(self, obs):
        return super().select_action(obs)

    def available_actions(self, obs) -> List[ResourceManagerActions]:
        available_actions = [ResourceManagerActions.NO_OP]

                

        HARVEST_MINERALS = 11
        COLLECT_GAS      = 12
        BUILD_REFINERY   = 13
        pass
