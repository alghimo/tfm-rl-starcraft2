from collections import deque, namedtuple
from typing import Any, Dict, List

import numpy as np

Sample = List[Any]
Buffer = namedtuple('Buffer', field_names=['state', 'action', 'action_args', 'reward', 'done', 'next_state'])


class ExperienceReplayBuffer:
    def __init__(self, memory_size: int = 50000, burn_in: int = 10000):
        self._memory_size = memory_size
        self._burn_in = burn_in
        self._buffer = Buffer
        self._replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size: int = 32) -> List[Sample]:
        """Sample a batch from the replay memory.

        Each sample is a list of 5 elements:
        - state: Current state
        - action: Action selected
        - reward: Reward obtained
        - done: Whether the action ended the episode
        - next_state: State of the environment after taking the action

        Args:
            batch_size (int, optional): Number of samples to take. Defaults to 32.

        Returns:
            List[Sample]: A batch, made of a list of batch_size samples.
        """
        samples = np.random.choice(len(self._replay_memory), batch_size,
                                   replace=False)

        batch = zip(*[self._replay_memory[i] for i in samples])

        return batch

    def append(self, state: Any, action: Any, action_args: Dict[str, Any], reward: float, done: bool, next_state: Any):
        """Add a new sample to the replay memory.

        Args:
            state (Any): Current state
            action (Any): Action selected
            action_args (Dict[str, Any]): Action Arguments selected
            reward (float): Reward obtained
            done (bool): Whether the episode ended
            next_state (Any): New state of the environment after taking the action
        """
        self._replay_memory.append(
            self._buffer(state, action, action_args, reward, done, next_state))

    @property
    def burn_in_capacity(self) -> float:
        """Calculates the percentage of the burn-in capacity that is filled.

        If the value is less than 1, it means that we still should keep filling
        the buffer before we start sampling.

        Returns:
            float: The percentage of burn-in capacity filled.
        """
        return len(self._replay_memory) / self._burn_in
