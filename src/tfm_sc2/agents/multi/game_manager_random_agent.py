import random
from typing import Any, Dict, Tuple

from pysc2.env.environment import TimeStep
from pysc2.lib import actions

from ...actions import AllActions
from ..base_agent import BaseAgent

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS
from .game_manager_base_agent import GameManagerBaseAgent


class GameManagerRandomAgent(GameManagerBaseAgent, BaseAgent):
    def select_action(self, obs: TimeStep, valid_actions = None) -> Tuple[AllActions, Dict[str, Any]]:
        if valid_actions is not None:
            action = random.choice(valid_actions)
        else:
            action = random.choice(self.agent_actions)

        return action
