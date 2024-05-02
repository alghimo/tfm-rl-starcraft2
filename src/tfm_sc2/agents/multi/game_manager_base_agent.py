
from pysc2.env.environment import TimeStep
from pysc2.lib import actions

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS
from ...actions import GameManagerActions
from .with_army_attack_manager_actions import WithArmyAttackManagerActions
from .with_army_recruit_manager_actions import WithArmyRecruitManagerActions
from .with_base_manager_actions import WithBaseManagerActions
from .with_game_manager_actions import WithGameManagerActions


class GameManagerBaseAgent(WithGameManagerActions):

    def __init__(self, base_manager: WithBaseManagerActions, army_recruit_manager: WithArmyRecruitManagerActions, attack_manager: WithArmyAttackManagerActions, **kwargs):
        super().__init__(**kwargs)
        self.base_manager = base_manager
        self._army_recruit_manager = army_recruit_manager
        self._attack_manager = attack_manager

    def forward_action(self, obs: TimeStep, action: GameManagerActions):
        match action:
            case GameManagerActions.EXPAND_BASE:
                proxy_manager = self._base_manager
            case GameManagerActions.EXPAND_ARMY:
                proxy_manager = self._army_recruit_manager
            case GameManagerActions.ATTACK:
                proxy_manager = self._army_attack_manager

        actual_action = proxy_manager.select_action(obs=obs)
        action_args, is_valid_action = proxy_manager._get_action_args(obs=obs, action=action)

        return actual_action, action_args, is_valid_action
