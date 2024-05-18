import random
from typing import Any, Dict, List, Tuple

from pysc2.env.environment import TimeStep
from pysc2.lib import actions, units

from ...actions import (
    AllActions,
    ArmyAttackManagerActions,
    ArmyRecruitManagerActions,
    BaseManagerActions,
    # ResourceManagerActions,
)
from ...constants import Constants
from ..dqn_agent import DQNAgent

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS
from ...types import Gas, Minerals


class SingleDQNAgent(DQNAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.__agent_actions = list(AllActions)
        self.__agent_actions = list(
            set(
                # list(ResourceManagerActions) +
                list(BaseManagerActions) +
                list(ArmyRecruitManagerActions) +
                list(ArmyAttackManagerActions)
            ))

    @property
    def agent_actions(self) -> List[AllActions]:
        return self.__agent_actions
