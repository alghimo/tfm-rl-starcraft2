import pickle
from collections import namedtuple
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, units
from typing_extensions import Self

from ..actions import AllActions
from ..constants import Constants, SC2Costs
from ..networks.dqn_network import DQNNetwork
from ..networks.experience_replay_buffer import ExperienceReplayBuffer
from ..types import AgentStage, DQNAgentParams, Gas, Minerals, RewardMethod, State
from .base_agent import BaseAgent
from .stats import AgentStats, AggregatedEpisodeStats, EpisodeStats


class DQNAgent(BaseAgent):
    _MAIN_NETWORK_FILE: str = "main_network.pt"
    _TARGET_NETWORK_FILE: str = "target_network.pt"
    # _AGENT_FILE: str = "agent.pkl"
    # _STATS_FILE: str =  "stats.parquet"

    def __init__(self, main_network: DQNNetwork,
                 hyperparams: DQNAgentParams,
                 target_network: DQNNetwork = None,
                 random_mode: bool = False,
                 **kwargs
                 ):
        """Deep Q-Network agent.

        Args:
            main_network (nn.Module): Main network
            buffer (ExperienceReplayBuffer): Memory buffer
            hyperparams (DQNAgentParams): Agent hyper parameters.
            target_network (nn.Module, optional): Target network. If not provided, then the main network will be cloned.
        """
        super().__init__(**kwargs)

        self.main_network = main_network
        self.target_network = target_network or deepcopy(main_network)
        self.hyperparams = hyperparams
        self.initial_epsilon = hyperparams.epsilon
        self.epsilon = hyperparams.epsilon
        self.loss = self.hyperparams.loss or torch.nn.HuberLoss()
        self._random_mode = random_mode
        self._is_burnin = self._buffer.burn_in_capacity < 1

        # Placeholders
        self._action_to_idx = None
        self._idx_to_action = None
        self._num_actions = None

        self.__prev_reward = None

        # Add some extra flags to the default ones (train / exploit)
        self._status_flags.update(dict(
            burnin_started=False,
            main_net_updated=False,
            target_net_updated=False,
        ))

        if self._checkpoint_path is not None:
            self._main_network_path = self._checkpoint_path / self._MAIN_NETWORK_FILE
            self._target_network_path = self._checkpoint_path / self._TARGET_NETWORK_FILE
        else:
            self._main_network_path = None
            self._target_network_path = None

    @property
    def _collect_stats(self) -> bool:
        return not (self._is_burnin or self._random_mode)

    def _update_checkpoint_paths(self):
        super()._update_checkpoint_paths()
        if self.checkpoint_path is None:
            self._main_network_path = None
            self._target_network_path = None
        else:
            self._main_network_path = self.checkpoint_path / self._MAIN_NETWORK_FILE
            self._target_network_path = self.checkpoint_path / self._TARGET_NETWORK_FILE

    @classmethod
    def _extract_init_arguments(cls, checkpoint_path: Path, agent_attrs: Dict[str, Any], map_name: str, map_config: Dict) -> Dict[str, Any]:
        parent_attrs = super()._extract_init_arguments(checkpoint_path=checkpoint_path, agent_attrs=agent_attrs, map_name=map_name, map_config=map_config)
        main_network_path = checkpoint_path / cls._MAIN_NETWORK_FILE
        target_network_path = checkpoint_path / cls._TARGET_NETWORK_FILE
        return dict(
            **parent_attrs,
            main_network=torch.load(main_network_path),
            target_network=torch.load(target_network_path),
            random_mode=agent_attrs.get("random_mode", agent_attrs.get("random_model", False)),
            hyperparams=agent_attrs["hyperparams"]
        )

    def _load_agent_attrs(self, agent_attrs: Dict):
        super()._load_agent_attrs(agent_attrs)
        self.initial_epsilon = agent_attrs["initial_epsilon"]
        self._num_actions = agent_attrs["num_actions"]
        self.loss = agent_attrs["loss"]
        if "epsilon" in agent_attrs:
            self.epsilon = agent_attrs["epsilon"]

    def _get_agent_attrs(self):
        parent_attrs = super()._get_agent_attrs()
        return dict(
            hyperparams=self.hyperparams,
            initial_epsilon=self.initial_epsilon,
            epsilon=self.epsilon,
            num_actions=self._num_actions,
            loss=self.loss,
            random_mode=self._random_mode,
            **parent_attrs
        )

    def save(self, checkpoint_path: Union[str|Path] = None):
        super().save(checkpoint_path=checkpoint_path)
        torch.save(self.main_network, self._main_network_path)
        torch.save(self.target_network, self._target_network_path)

    def _current_agent_stage(self):
        if self._is_burnin:
            return AgentStage.BURN_IN
        if self._exploit:
            return AgentStage.EXPLOIT
        if self.is_training:
            return AgentStage.TRAINING
        return AgentStage.UNKNOWN

    @property
    def memory_replay_ready(self) -> bool:
        return self._buffer.burn_in_capacity >= 1

    def reset(self, **kwargs):
        """Initialize the agent."""
        super().reset(**kwargs)
        # Last observation
        self.__prev_reward = None

        self._is_burnin = self._buffer.burn_in_capacity < 1
        self._current_episode_stats.is_burnin = self._is_burnin

    @property
    def is_training(self):
        return super().is_training and (not self._random_mode) and (not self._is_burnin)


    def select_action(self, obs: TimeStep) -> Tuple[AllActions, Dict[str, Any]]:
        # available_actions = self.available_actions(obs)
        # self.logger.debug(f"Available actions: {available_actions}")
        available_actions = [a for a in self.agent_actions if a in self._map_config["available_actions"]]
        # if len(available_actions) > 1 and AllActions.NO_OP in available_actions:
        #     available_actions = [a for a in available_actions if a != AllActions.NO_OP]
        # One-hot encoded version of available actions
        valid_actions = self._actions_to_network(available_actions)
        if not any(valid_actions):
            valid_actions = None
        if (not self._exploit) and (self._is_burnin or self._random_mode):
            if self._random_mode:
                self.logger.debug(f"Random mode - collecting experience from random actions")
            elif not self._status_flags["burnin_started"]:
                self.logger.info(f"Starting burn-in")
                self._status_flags["burnin_started"] = True
            else:
                self.logger.debug(f"Burn in capacity: {100 * self._buffer.burn_in_capacity:.2f}%")

            raw_action = self.main_network.get_random_action(valid_actions=valid_actions)
            # raw_action = self.main_network.get_random_action()
        elif self.is_training:
            if not self._status_flags["train_started"]:
                self.logger.info(f"Starting training")
                self._status_flags["train_started"] = True
            raw_action = self.main_network.get_action(self._current_state_tensor, epsilon=self.epsilon, valid_actions=valid_actions)
            # raw_action = self.main_network.get_action(self.__current_state, epsilon=self.epsilon)
        else:
            if not self._status_flags["exploit_started"]:
                self.logger.info(f"Starting exploit")
                self._status_flags["exploit_started"] = True

            available_actions = self.available_actions(obs)
            valid_actions = self._actions_to_network(available_actions)
            # available_actions = [a for a in self.agent_actions if a in self._map_config["available_actions"]]
            # One-hot encoded version of available actions
            # When exploiting, do not use the invalid action masking
            # raw_action = self.main_network.get_greedy_action(self.__current_state)
            raw_action = self.main_network.get_greedy_action(self._current_state_tensor, valid_actions=valid_actions)

        # Convert the "raw" action to a the right type of action
        action = self._idx_to_action[raw_action]

        action_args, is_valid_action = self._get_action_args(obs=obs, action=action)

        return action, action_args, is_valid_action

    def pre_step(self, obs: TimeStep):
        super().pre_step(obs)
        if not obs.first():
            # do updates
            done = obs.last()
            # self._buffer.append(self._prev_state, self._prev_action, self._prev_action_args, self._current_reward, self._current_adjusted_reward, self._current_score, done, self._current_state)

            if (not self._is_burnin) and self.is_training:
                main_net_updated = False
                target_net_updated = False
                if self.hyperparams.main_network_update_frequency > -1:
                    if (self._current_episode_stats.steps % self.hyperparams.main_network_update_frequency) == 0:
                        if not self._status_flags["main_net_updated"]:
                            self.logger.info(f"First main network update")
                            self._status_flags["main_net_updated"] = True
                        self._current_episode_stats.losses = self.update_main_network(self._current_episode_stats.losses)
                        main_net_updated = True
                if self.hyperparams.target_network_sync_frequency > -1:
                    if (self._current_episode_stats.steps % self.hyperparams.target_network_sync_frequency) == 0:
                        if not self._status_flags["target_net_updated"]:
                            self.logger.info(f"First target network update")
                            self._status_flags["target_net_updated"] = True
                        self.synchronize_target_network()
                        target_net_updated = True
                # HERE

                if done:
                    if not main_net_updated:
                        # If we finished but didn't update, perform one last update
                        self._current_episode_stats.losses = self.update_main_network(self._current_episode_stats.losses)
                    if not target_net_updated:
                        self.synchronize_target_network()

                    self._current_episode_stats.epsilon = self.epsilon if not self._random_mode else 1.
                    self._current_episode_stats.loss = np.mean(self._current_episode_stats.losses)
                    self.epsilon = max(self.epsilon * self.hyperparams.epsilon_decay, self.hyperparams.min_epsilon)
            elif done and self._is_burnin:
                self._current_episode_stats.epsilon = 1.
            elif done and self._exploit:
                self._current_episode_stats.epsilon = 0.

    def post_step(self, obs: TimeStep, action: AllActions, action_args: Dict[str, Any], original_action: AllActions, original_action_args: Dict[str, Any], is_valid_action: bool):
        super().post_step(obs, action, action_args, original_action, original_action_args, is_valid_action)

        if obs.last():
            self._current_episode_stats.is_random_mode = self._random_mode

    def _get_end_of_episode_info_components(self) -> List[str]:
        return super()._get_end_of_episode_info_components() + [
            f"Epsilon {self._current_episode_stats.epsilon:.4f}",
            f"Mean Loss ({len(self._current_episode_stats.losses)} updates) {self._current_episode_stats.mean_loss:.4f}",
        ]

    def _calculate_loss(self, batch: Iterable[Tuple]) -> torch.Tensor:
        """Calculate the loss for a batch.

        Args:
            batch (Iterable[Tuple]): Batch to calculate the loss on.

        Returns:
            torch.Tensor: The calculated loss between the calculated and the predicted values.
        """

        # Convert elements from the replay buffer to tensors
        states, actions, action_args, rewards, adjusted_rewards, scores, dones, next_states = [i for i in batch]
        states = torch.stack(states).to(device=self.device)
        next_states = torch.stack(next_states).to(device=self.device)
        match self._reward_method:
            case RewardMethod.SCORE:
                rewards_to_use = scores
            case RewardMethod.ADJUSTED_REWARD:
                rewards_to_use = adjusted_rewards
            case RewardMethod.REWARD:
                rewards_to_use = rewards
            case _:
                self.logger.warning(f"Unknown reward method {self._reward_method.name} - using default rewards")
                rewards_to_use = rewards
        rewards_vals = torch.FloatTensor(rewards_to_use).to(device=self.device)
        actions_vals = torch.LongTensor(np.array(actions)).to(device=self.device).reshape(-1,1)
        dones_t = torch.BoolTensor(dones).to(device=self.device)

        # Get q-values from the main network
        qvals = torch.gather(self.main_network.get_qvals(states), 1, actions_vals)
        # Get q-values from the target network
        qvals_next = torch.max(self.target_network.get_qvals(next_states), dim=-1)[0].detach()
        # Set terminal states to 0
        qvals_next[dones_t] = 0

        # Get expected q-values
        expected_qvals = self.hyperparams.gamma * qvals_next + rewards_vals

        return self.loss(qvals, expected_qvals.reshape(-1,1))

    def update_main_network(self, episode_losses: List[float] = None) -> List[float]:
        """Update the main network.

        Normally we only perform the update every certain number of steps, defined by
        main_network_update_frequency, but if force_update is set to true, then
        the update will happen, independently of the current step count.

        Args:
            episode_losses (List, optional): List with episode losses.
        Returns:
            List[float]: A list with all the losses in the current episode.
        """
        episode_losses = episode_losses or []

        self.main_network.optimizer.zero_grad()  # Remove previous gradients
        batch = self._buffer.sample_batch(batch_size=self.hyperparams.batch_size) # Sample experiences from the buffer

        loss = self._calculate_loss(batch)# Get batch loss
        loss.backward() # Backward pass to get gradients
        self.main_network.optimizer.step()
        if hasattr(self.main_network, "scheduler") and (self.main_network.scheduler is not None):
            self.main_network.scheduler.step() # Apply the gradients to the main network

        if self.device == 'cuda':
            loss = loss.detach().cpu().numpy()
        else:
            loss = loss.detach().numpy()

        episode_losses.append(float(loss))
        return episode_losses

    def synchronize_target_network(self):
        """Synchronize the target network with the main network parameters.

        When the target_sync_mode is set to "soft", a soft update is made, so instead of overwriting
        the target fully, we update it by mixing in the current target parameters and the main network
        parameters. In practice, we keep a fraction (1 - update_tau) from the target network, and add
        to it a fraction update_tau from the main network.
        """

        if self.hyperparams.target_sync_mode == "soft":
            for target_var, var in zip(self.target_network.parameters(), self.main_network.parameters()):
                    target_var.data.copy_((1. - self.hyperparams.update_tau) * target_var.data + (self.hyperparams.update_tau) * var.data)
        else:
            self.target_network.load_state_dict(self.main_network.state_dict())
