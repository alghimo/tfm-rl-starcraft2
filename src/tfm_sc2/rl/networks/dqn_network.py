from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from gymnasium.core import Env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class DQNNetwork(nn.Module, ABC):
    """Base class for Deep Q-Networks on Gym environments with discrete actions.

    This class makes no assumptions over the type of network, and it will
    also not check that the input/output features of the network layers
    matches the shape of the observation space / number of actions.

    Args:
        env (Env): Gym environment
        model (nn.Module): The network that will be used
        optimizer (optim.Optimizer): Optimizer to use
    """
    def __init__(self, model_layers: List[nn.Module], learning_rate: float):
        super().__init__()
        
        self.input_shape = None #env.observation_space.shape[0]
        self.n_outputs = None #env.action_space.n
        self.actions = list(range(self.n_outputs))
        model = torch.nn.Sequential(*model_layers)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.model = model
        self.optimizer = optimizer
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def get_random_action(self) -> int:
        """Select a random action.

        Returns:
            int: Selected action
        """
        return np.random.choice(self.actions)

    def get_action(self, state: Union[np.ndarray, list, tuple], epsilon: float = 0.05) -> int:
        """Select an action using epsilon-greedy method.

        Args:
            state (Union[np.ndarray, list, tuple]): Current state.
            epsilon (float, optional): Probability of taking a random action. Defaults to 0.05.

        Returns:
            int: Action selected.
        """
        if np.random.random() < epsilon:
            action = self.get_random_action()
        else:
            qvals = self.get_qvals(state)
            # Action selected = index of the highest Q-value
            action = torch.max(qvals, dim=-1)[1].item()

        return action

    def get_qvals(self, state: torch.Tensor) -> torch.Tensor:
        """Get the Q-values for a certain state.

        Args:
            state (torch.Tensor): State to get the q-values for.

        Returns:
            torch.Tensor: Tensor with the Q-value for each of the actions.
        """
        out = self.model(state)

        return out
