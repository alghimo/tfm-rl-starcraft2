from ..base_agent import BaseAgent
from .with_army_recruit_manager_actions import WithArmyRecruitManagerActions
from .with_random_action_selection import WithRandomActionSelection


class ArmyRecruitManagerRandomAgent(WithArmyRecruitManagerActions, WithRandomActionSelection, BaseAgent):
    pass