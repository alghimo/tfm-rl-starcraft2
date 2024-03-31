from copy import deepcopy, copy
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_agent import BaseAgent
from ..memory.experience_replay_buffer import ExperienceReplayBuffer
from ..networks.dqn_network import DQNNetwork


class DQNAgent(BaseAgent):
    def __init__(self, main_network: DQNNetwork, buffer: ExperienceReplayBuffer,
                 solve_reward_threshold: float, solve_num_episodes: int = 100, target_network: DQNNetwork = None,
                 epsilon: float = 0.1, eps_decay: float = 0.99, min_epsilon: float = 0.01,
                 batch_size: int = 32, gamma=0.99,
                 loss: nn.Module = None,
                 main_network_update_frequency: int = 1, target_network_sync_frequency: int = 50,
                 target_sync_mode = "full", update_tau: bool = 0.001,
                 keep_training_after_solving: bool = False, restore_to_best_version: bool = False
                 ):
        """Deep Q-Network agent.

        Args:
            main_network (nn.Module): Main network
            buffer (ExperienceReplayBuffer): Memory buffer
            solve_reward_threshold (float): Minimum reward to solve the environment
            solve_num_episodes (int, optional): Minimum number of episodes where we need to get the minimum reward to solve the environment. Defaults to 100.
            target_network (nn.Module, optional): Target network. If not provided, then the main network will be cloned.
            epsilon (float, optional): Initial value of epsilon for the epsilon-greedy action selection. Defaults to 0.1.
            eps_decay (float, optional): Exponencial decay coefficient for epsilon. Defaults to 0.99.
            batch_size (int, optional): Batch size. Defaults to 32.
            loss (_type_, optional): Loss function. Defaults to torch.nn.HuberLoss.
        """

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.main_network = main_network
        self.target_network = target_network or deepcopy(main_network) # Default the target network to a copy of the main network
        self.buffer = buffer
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        
        # We will consider that the agent beat the environment if it gets a reward greater than
        # 'reward_threshold' for 'solve_num_episodes' consecutive episodes
        self.solve_num_episodes = solve_num_episodes
        self.solve_reward_threshold = solve_reward_threshold
        self.main_network_update_frequency = main_network_update_frequency
        self.target_network_sync_frequency = target_network_sync_frequency
        self.target_sync_mode = target_sync_mode
        self.update_tau = update_tau
        self.keep_training_after_solving = keep_training_after_solving
        self.restore_to_best_version = restore_to_best_version
        self.initialize()
        self.loss = loss or torch.nn.HuberLoss()

    def initialize(self):
        """Initialize the agent."""

        self.step_count = 0
        self.state = self.env.reset()[0]

        self.update_loss = []
        self.training_rewards = []
        self.episode_eps = []
        self.mean_rewards = []
        self.mean_rewards_10 = []
        self.episode_losses = []
        self.episode_steps = []
        self.mean_steps = []
        self.max_episode_rewards = None

    def take_action(self, action: int) -> Tuple[bool, float]:
        """Take an action in the environment.

        If the episode is finished after taking the action, the environment is reset.

        Args:
            action (int): Action to take

        Returns:
            Tuple[bool, float]: A bool indicating if the episode is finished, and a float with the reward of the step.
        """

        new_state, reward, done, truncated, _ = self.env.step(action)
        done = done or truncated

        if not done:
            self.buffer.append(self.state, action, reward, done, new_state)
            self.state = new_state

        if done:
            self.state = self.env.reset()[0]

        return done, reward

    def take_step(self, eps: float) -> Tuple[bool, float]:
        """Perform a step in the environment.

        The action will be selected from the main network, and will have a probability
        "eps" of taking a random action.

        Args:
            eps (float): Probability of taking a random action

        Returns:
            Tuple[bool, float]: A bool indicating if the episode is finished, and a float with the reward of the step.
        """
        action = self.main_network.get_action(self.state, eps)
        done, reward = self.take_action(action)

        return done, reward

    def play_episode(self, episode_number: int, mode="train") -> Tuple[int, float, float]:
        """Play a full episode.

        If mode is "train", extra metrics are captured, to be used later.

        Args:
            episode_number (int): Episode number, only used for informative purposes.
            mode (str): Set to "train" during training

        Returns:
            Tuple[int, float, float]: A tuple with (number of steps, episode reward, mean rewards of last 100 episodes)
        """

        self.state = self.env.reset()[0]
        num_steps = 0
        episode_reward = 0
        done = False
        is_training = mode == "train"
        mean_rewards = None
        episode_losses = []

        # In not training, use an epsilon of 0
        eps = self.epsilon if is_training else 0
        while not done:
            done, reward = self.take_step(eps)
            self.step_count += 1

            num_steps += 1
            episode_reward += reward

            if is_training:
                episode_losses = self.update_main_network(episode_losses)
                self.synchronize_target_network()

        if is_training:
            # Add the episode rewards to the training rewards
            self.training_rewards.append(episode_reward)
            # Register the epsilon used in the last episode played
            self.episode_eps.append(self.epsilon)
            # We'll use the average loss as the episode loss
            if any(episode_losses):
                episode_loss = sum(episode_losses) / len(episode_losses)
            else:
                episode_loss = 100
            self.episode_losses.append(episode_loss)
            self.episode_steps.append(num_steps)
            """
            Get the average reward of the last episodes. Here we keep track
            of the rolling average of the las N episodes, where N is the minimum number
            of episodes we need to solve the environment.
            """
            mean_rewards = np.mean(self.training_rewards[-self.solve_num_episodes:])
            # Also keep the rolling average of the last 10 episodes
            mean_rewards_10 = np.mean(self.training_rewards[-10:])
            self.mean_rewards.append(mean_rewards)
            self.mean_rewards_10.append(mean_rewards_10)

            mean_steps = np.mean(self.episode_steps[-self.solve_num_episodes:])
            self.mean_steps.append(mean_steps)

            # Check if we have a new max score
            if self.max_episode_rewards is None or (episode_reward > self.max_episode_rewards):
                self.max_episode_rewards = episode_reward

            print(
                f"\rEpisode {episode_number} :: "
                + f"Mean Rewards ({self.solve_num_episodes} ep) {mean_rewards:.2f} :: "
                + f"Mean Rewards (10ep) {mean_rewards_10:.2f} :: "
                + f"Epsilon {self.epsilon:.4f} :: "
                + f"Maxim {self.max_episode_rewards:.2f} :: "
                + f"Steps: {num_steps}\t\t\t\t", end="")

        self.env.close()

        return num_steps, episode_reward, mean_rewards

    def train(self, max_episodes: int):
        """Train the agent for a certain number of episodes.

        Depending on the settings, the agent might stop training as soon as the environment is solved,
        or it might continue training despite having solved it.

        Args:
            max_episodes (int): Maximum number of episodes to play.
        """

        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity < 1:
            action = self.main_network.get_random_action()
            self.take_action(action)

        episode = 0
        training = True
        print("Training...")
        max_reward = None

        best_main_net = deepcopy(self.main_network)
        best_target_net = deepcopy(self.target_network)
        best_episode = 0
        best_mean_rewards = -10000

        solved = False
        self.epsilon = self.initial_epsilon

        while training:
            # Play an episode
            episode_steps, episode_reward, mean_rewards = self.play_episode(episode_number=episode)
            episode += 1

            # If we reached the maximum number of episodes, finish the training
            if episode >= max_episodes:
                training = False
                print("\nEpisode limit reached.")
                break

            min_episodes_reached = self.solve_num_episodes <  episode
            new_best_mean_reward = mean_rewards > best_mean_rewards
            solve_threshold_reached = mean_rewards >= self.solve_reward_threshold

            # Once the minimum number of episodes is reached, keep track of the best model
            if min_episodes_reached and new_best_mean_reward:
                best_main_net.load_state_dict(self.main_network.state_dict())
                best_target_net.load_state_dict(self.target_network.state_dict())
                best_episode = episode
                best_mean_rewards = mean_rewards

            # Check if the environment has been solved
            if solve_threshold_reached and min_episodes_reached:
                # We only trigger this once, in case we keep training after having solved the environment
                if not solved:
                    print(f'\nEnvironment solved in {episode} episodes!')
                    solved = True

                # On the other hand, if the environment is solved and we should not continue training
                # after solving, we are done training.
                if not self.keep_training_after_solving:
                    training = False
                    break

            self.epsilon = max(self.epsilon * self.eps_decay, self.min_epsilon)

        # If keep_training_after_solving is true, we restore the agent to the version
        # that achieved the best mean rewards
        if self.restore_to_best_version:
            print(f"\nFinished training after {episode} episodes, restoring agent to best version at episode {best_episode}")
            print(f"Best agent got mean rewards {best_mean_rewards:.2f} at episode {best_episode}")

            self.main_network.load_state_dict(best_main_net.state_dict())
            self.target_network.load_state_dict(best_target_net.state_dict())
            self.training_rewards = self.training_rewards[:best_episode]
            self.mean_rewards = self.mean_rewards[:best_episode]
            self.mean_rewards_10 = self.mean_rewards_10[:best_episode]
            self.episode_eps = self.episode_eps[:best_episode]
            self.episode_losses = self.episode_losses[:best_episode]
            self.episode_steps = self.episode_steps[:best_episode]

    def calculate_loss(self, batch: Iterable[Tuple]) -> torch.Tensor:
        """Calculate the loss for a batch.

        Args:
            batch (Iterable[Tuple]): Batch to calculate the loss on.

        Returns:
            torch.Tensor: The calculated loss between the calculated and the predicted values.
        """

        # Separem les variables de l'experiència i les convertim a tensors
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_vals = torch.FloatTensor(rewards).to(device=self.device)
        actions_vals = torch.LongTensor(np.array(actions)).to(device=self.device).reshape(-1,1)
        dones_t = torch.ByteTensor(dones).to(device=self.device)

        # Obtenim els valors de Q de la xarxa principal
        qvals = torch.gather(self.main_network.get_qvals(states), 1, actions_vals)
        # Obtenim els valors de Q de la xarxa objectiu
        # El paràmetre detach() evita que aquests valors actualitzin la xarxa objectiu
        qvals_next = torch.max(self.target_network.get_qvals(next_states),
                               dim=-1)[0].detach()
        qvals_next[dones_t] = 0 # 0 en estats terminals

        expected_qvals = self.gamma * qvals_next + rewards_vals

        return self.loss(qvals, expected_qvals.reshape(-1,1))

    def update_main_network(self, episode_losses: List[float] = None, force_update: bool = False) -> List[float]:
        """Update the main network.

        Normally we only perform the update every certain number of steps, defined by
        main_network_update_frequency, but if force_update is set to true, then
        the update will happen, independently of the current step count.

        Args:
            force_update (bool, optional): Force update of the network without checking the step count. Defaults to False.
        Returns:
            List[float]: A list with all the losses in the current episode.
        """
        episode_losses = episode_losses or []
        if (self.step_count % self.main_network_update_frequency != 0) and not force_update:
            return episode_losses

        self.main_network.optimizer.zero_grad()  # eliminem qualsevol gradient passat
        batch = self.buffer.sample_batch(batch_size=self.batch_size) # seleccionem un conjunt del buffer

        loss = self.calculate_loss(batch)# calculem la pèrdua
        loss.backward() # calculem la diferència per obtenir els gradients
        self.main_network.optimizer.step() # apliquem els gradients a la xarxa neuronal

        if self.device == 'cuda':
            loss = loss.detach().cpu().numpy()
        else:
            loss = loss.detach().numpy()

        self.update_loss.append(loss)
        episode_losses.append(float(loss))
        return episode_losses

    def synchronize_target_network(self, force_update: bool = False):
        """Synchronize the target network with the main network parameters.

        When the target_sync_mode is set to "soft", a soft update is made, so instead of overwriting
        the target fully, we update it by mixing in the current target parameters and the main network
        parameters. In practice, we keep a fraction (1 - update_tau) from the target network, and add
        to it a fraction update_tau from the main network.

        Args:
            force_update (bool, optional): If true, the target network will be synched, no matter the step count. Defaults to False.
        """
        if (self.step_count % self.target_network_sync_frequency != 0) and not force_update:
            return

        if self.target_sync_mode == "soft":
            for target_var, var in zip(self.target_network.parameters(), self.main_network.parameters()):
                    target_var.data.copy_((1. - self.update_tau) * target_var.data + (self.update_tau) * var.data)
        else:
            self.target_network.load_state_dict(self.main_network.state_dict())
