from typing import List

from pysc2.env.environment import TimeStep

from ..base_agent import BaseAgent
from ...actions import AllActions


class SingleRandomAgent(BaseAgent):
    @property
    def agent_actions(self) -> List[AllActions]:
        return list(AllActions)
    
    def select_action(self, obs: TimeStep):
        pass