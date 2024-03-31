from typing import List

from absl import app
import numpy as np
from pysc2.env import sc2_env
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, features

from .agent_actions import GameManagerActions
from ..base_agent import BaseAgent

class GameManager(BaseAgent):
    def __init__(self, resource_manager: BaseAgent, base_manager: BaseAgent, army_recruit_manager: BaseAgent, army_attack_manager: BaseAgent, **kwargs):
        super().__init__(**kwargs)
        self._resource_manager = resource_manager
        self._base_manager = base_manager
        self._army_recruit_manager = army_recruit_manager
        self._army_attack_manager = army_attack_manager
        self._idx_to_action = {idx: a for (idx, a) in enumerate(list(GameManagerActions))}
        self._action_to_idx = {a: idx for (idx, a) in enumerate(list(GameManagerActions))}

    def actions(self) -> List[GameManagerActions]:
        return list(self._idx_to_action.values())

    def action_to_idx(self, action: GameManagerActions) -> int:
        return self._action_to_idx[action]

    def action_to_idx(self, action: int) -> GameManagerActions:
        return self._idx_to_action[action]
    
    def preprocess_observation(self, obs: TimeStep):
        return super().preprocess_observation(obs)
    
    def select_action(self, obs):
        return super().select_action(obs)
    
    def step(self, obs: TimeStep):
        super().step(obs)

        import pdb
        pdb.set_trace()

        return actions.FUNCTIONS.no_op()
