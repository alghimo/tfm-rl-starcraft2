from typing import Dict

from pysc2.env.environment import TimeStep
from pysc2.lib import actions, features, units

from ....types import ScvState
from ..agent_utils import AgentUtils


class ScvManager(AgentUtils):
    def __init__(self) -> None:
        self._init_units()
    
    def reset(self):
        self._init_units()

    def _init_units(self):
        self._scvs: Dict[int, features.FeatureUnit] = {}
        self._scv_states: Dict[int, ScvState] = {}

    def update(self, obs: TimeStep):
        all_scvs = self.get_self_units(obs, units.Terran.SCV)

        for scv in all_scvs:
            if scv.tag not in self._scvs:
                self.logger.info(f"New SCV: (#{scv.tag}#)")
                self._scvs[scv.tag] = scv

        