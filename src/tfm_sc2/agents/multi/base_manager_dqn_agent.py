from ..dqn_agent import DQNAgent
from .with_base_manager_actions import WithBaseManagerActions


class BaseManagerDQNAgent(WithBaseManagerActions, DQNAgent):
    pass