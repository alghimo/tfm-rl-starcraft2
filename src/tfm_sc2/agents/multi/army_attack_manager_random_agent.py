from ..base_agent import BaseAgent
from .with_army_attack_manager_actions import WithArmyAttackManagerActions
from .with_random_action_selection import WithRandomActionSelection


class ArmyAttackManagerRandomAgent(WithArmyAttackManagerActions, WithRandomActionSelection, BaseAgent):
    pass