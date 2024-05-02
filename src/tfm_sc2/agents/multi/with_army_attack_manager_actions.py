from typing import List

from ...actions import AllActions, ArmyAttackManagerActions


class WithArmyAttackManagerActions:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__agent_actions = list(ArmyAttackManagerActions)

    @property
    def agent_actions(self) -> List[AllActions]:
        return self.__agent_actions
