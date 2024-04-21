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
        action_args, is_valid_action = self._get_action_args(obs=obs, action=action)

        if not is_valid_action:
            self.logger.warning(f"Action {action.name} is not valid anymore, returning NO_OP")
            return AllActions.NO_OP, None

        return action, action_args
