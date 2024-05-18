from ..base_agent import BaseAgent
from .with_base_manager_actions import WithBaseManagerActions
from .with_random_action_selection import WithRandomActionSelection


class BaseManagerRandomAgent(WithBaseManagerActions, WithRandomActionSelection, BaseAgent):
    pass