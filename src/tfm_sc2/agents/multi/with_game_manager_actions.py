from typing import List

from ...actions import AllActions, GameManagerActions


class WithGameManagerActions:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__agent_actions = list(GameManagerActions)

    @property
    def agent_actions(self) -> List[AllActions]:
        return self.__agent_actions
