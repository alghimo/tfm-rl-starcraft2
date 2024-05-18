import random
from typing import Any, Dict, Tuple

from pysc2.env.environment import TimeStep
from pysc2.lib import actions

from ...actions import AllActions
from ..dqn_agent import DQNAgent

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS
from ...actions import GameManagerActions
from .game_manager_base_agent import GameManagerBaseAgent


class GameManagerDQNAgent(GameManagerBaseAgent, DQNAgent):

    def _select_game_manager_action(self, obs: TimeStep) -> GameManagerActions:
        return random.choice(self.agent_actions)

    def select_action(self, obs: TimeStep, valid_actions = None) -> Tuple[AllActions, Dict[str, Any]]:
        if valid_actions is not None:
            valid_actions = self._actions_to_network(valid_actions)
        if (self._random_mode) or (self._train and (self._buffer.burn_in_capacity < 1)):
            if not self._status_flags["burnin_started"]:
                self.logger.info(f"Starting burn-in")
                self._status_flags["burnin_started"] = True
            if self._random_mode:
                self.logger.debug(f"Random mode - collecting experience from random actions")
            else:
                self.logger.debug(f"Burn in capacity: {100 * self._buffer.burn_in_capacity:.2f}%")
            raw_action = self.main_network.get_random_action(valid_actions=valid_actions)
            # raw_action = self.main_network.get_random_action()
        elif self.is_training:
            if not self._status_flags["train_started"]:
                self.logger.info(f"Starting training")
                self._status_flags["train_started"] = True
            raw_action = self.main_network.get_action(self._current_state_tensor, epsilon=self.epsilon, valid_actions=valid_actions)
        else:
            if not self._status_flags["exploit_started"]:
                self.logger.info(f"Starting exploit")
                self._status_flags["exploit_started"] = True
            raw_action = self.main_network.get_greedy_action(self._current_state_tensor, valid_actions=valid_actions)

        # Convert the "raw" action to a the right type of action
        action = self._idx_to_action[raw_action]

        return action
