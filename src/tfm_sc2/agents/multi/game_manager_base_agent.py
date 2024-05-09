
from pysc2.env.environment import TimeStep
from pysc2.lib import actions

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS
from ...actions import AllActions, GameManagerActions
from .with_army_attack_manager_actions import WithArmyAttackManagerActions
from .with_army_recruit_manager_actions import WithArmyRecruitManagerActions
from .with_base_manager_actions import WithBaseManagerActions
from .with_game_manager_actions import WithGameManagerActions


class GameManagerBaseAgent(WithGameManagerActions):

    def __init__(self, base_manager: WithBaseManagerActions, army_recruit_manager: WithArmyRecruitManagerActions, army_attack_manager: WithArmyAttackManagerActions, **kwargs):
        super().__init__(**kwargs)
        self._base_manager = base_manager
        self._army_recruit_manager = army_recruit_manager
        self._army_attack_manager = army_attack_manager

        self._base_manager.setup_actions()
        self._army_recruit_manager.setup_actions()
        self._army_attack_manager.setup_actions()

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

        return actual_action, action_args, is_valid_action, proxy_manager

    def pre_step(self, obs: TimeStep):
        super().pre_step(obs)
        self._base_manager.pre_step(obs)
        self._army_recruit_manager.pre_step(obs)
        self._army_attack_manager.pre_step(obs)

    def step(self, obs: TimeStep):
        if obs.first():
            self._setup_positions(obs)
            self._base_manager._setup_positions(obs)
            self._army_recruit_manager._setup_positions(obs)
            self._army_attack_manager._setup_positions(obs)
        # import pdb
        # pdb.set_trace()
        self.pre_step(obs)

        super().step(obs, only_super_step=True)

        self.update_supply_depot_positions(obs)
        self._base_manager.update_supply_depot_positions(obs)
        self._army_recruit_manager.update_supply_depot_positions(obs)
        self._army_attack_manager.update_supply_depot_positions(obs)
        self.update_command_center_positions(obs)
        self._base_manager.update_supply_depot_positions(obs)
        self._army_recruit_manager.update_supply_depot_positions(obs)
        self._army_attack_manager.update_supply_depot_positions(obs)
        self.update_barracks_positions(obs)
        self._base_manager.update_barracks_positions(obs)
        self._army_recruit_manager.update_barracks_positions(obs)
        self._army_attack_manager.update_barracks_positions(obs)

        game_manager_action = self.select_action(obs)

        action, action_args, is_valid_action, proxy_manager = self.forward_action(obs=obs, action=game_manager_action)
        original_action = action
        if not is_valid_action:
            self.logger.debug(f"Sub-agent action {action.name} is not valid anymore, returning NO_OP")
            action = AllActions.NO_OP
            action_args = None
            self._current_episode_stats.add_invalid_action(game_manager_action)
        else:
            self._barrack_positions = proxy_manager._barrack_positions
            self._supply_depot_positions = proxy_manager._supply_depot_positions
            self._command_center_positions = proxy_manager._command_center_positions
            self._current_episode_stats.add_valid_action(game_manager_action)
            self.logger.debug(f"[Step {self.steps}] Manager action: {game_manager_action.name} // Sub-agent action {action.name} (original action = {original_action})")

        self.post_step(obs, game_manager_action, None, game_manager_action, None, True)

        game_action = self._action_to_game[action]

        self.post_step(obs, game_manager_action, None, game_manager_action, None, is_valid_action)

        if action_args is not None:
            return game_action(**action_args)

        return game_action()
