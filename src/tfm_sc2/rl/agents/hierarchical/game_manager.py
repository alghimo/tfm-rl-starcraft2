from ..base_agent import BaseAgent


class GameManager(BaseAgent):
    def __init__(self, resource_manager: BaseAgent, base_manager: BaseAgent, army_recruit_manager: BaseAgent, army_attack_manager: BaseAgent, **kwargs):
        super().__init__(**kwargs)
        self._resource_manager = resource_manager
        self._base_manager = base_manager
        self._army_recruit_manager = army_recruit_manager
        self._army_attack_manager = army_attack_manager
