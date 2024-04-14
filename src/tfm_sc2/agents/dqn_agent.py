from collections import namedtuple
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, units

from ..actions import AllActions
from ..constants import Constants
from ..networks.dqn_network import DQNNetwork
from ..networks.experience_replay_buffer import ExperienceReplayBuffer
from ..types import Gas, Minerals
from .base_agent import BaseAgent

DQNAgentParams = namedtuple('DQNAgentParams',
                            field_names=["epsilon", "epsilon_decay", "min_epsilon", "batch_size", "gamma", "loss", "main_network_update_frequency", "target_network_sync_frequency", "target_sync_mode", "update_tau"],
                            defaults=(0.1, 0.99, 0.01, 32, 0.99, None, 1, 50, "soft", 0.001))

State = namedtuple('State',
                            field_names=[
                                # Command centers
                                "num_command_centers", "num_completed_command_centers",
                                "command_center_0_order_length", "command_center_1_order_length", "command_center_2_order_length", "command_center_3_order_length",
                                "command_center_0_num_workers", "command_center_1_num_workers", "command_center_2_num_workers", "command_center_3_num_workers",
                                # Workers
                                "num_workers", "num_idle_workers", "pct_idle_workers", "num_mineral_harvesters", "pct_mineral_harvesters", "num_gas_harvesters",  "pct_gas_harvesters",
                                # Buildings
                                "num_refineries", "num_completed_refineries", "num_supply_depots", "num_completed_supply_depots", "num_barracks", "num_completed_barracks",
                                # Army
                                "num_marines", "num_marines_in_queue", "num_other_army_units",
                                # Resources
                                "free_supply", "minerals", "gas",
                                # Scores
                                # Cumulative
                                "score_cumulative_score", "score_cumulative_idle_production_time", "score_cumulative_idle_worker_time",
                                "score_cumulative_total_value_units", "score_cumulative_total_value_structures", "score_cumulative_killed_value_units",
                                "score_cumulative_killed_value_structures", "score_cumulative_collected_minerals", "score_cumulative_collected_vespene",
                                "score_cumulative_collection_rate_minerals", "score_cumulative_collection_rate_vespene", "score_cumulative_spent_minerals",
                                "score_cumulative_spent_vespene",
                                # By category
                                "score_food_used_none", "score_food_used_army", "score_food_used_economy", "score_food_used_technology", "score_food_used_upgrade", "score_by_vital",
                                "score_killed_minerals_none", "score_killed_minerals_army", "score_killed_minerals_economy", "score_killed_minerals_technology", "score_killed_minerals_upgrade", "score_killed_minerals_none",
                                "score_killed_vespene_army", "score_killed_vespene_economy", "score_killed_vespene_technology", "score_killed_vespene_upgrade", "score_killed_vespene_none",
                                "score_lost_minerals_none", "score_lost_minerals_army", "score_lost_minerals_economy", "score_lost_minerals_technology", "score_lost_minerals_upgrade", "score_lost_minerals_none",
                                "score_lost_vespene_army", "score_lost_vespene_economy", "score_lost_vespene_technology", "score_lost_vespene_upgrade", "score_lost_vespene_none",
                                "score_friendly_fire_minerals_none", "score_friendly_fire_minerals_army", "score_friendly_fire_minerals_economy", "score_friendly_fire_minerals_technology", "score_friendly_fire_minerals_upgrade",
                                "score_friendly_fire_vespene_none", "score_friendly_fire_vespene_army", "score_friendly_fire_vespene_economy", "score_friendly_fire_vespene_technology", "score_friendly_fire_vespene_upgrade",
                                "score_used_minerals_none", "score_used_minerals_army", "score_used_minerals_economy", "score_used_minerals_technology", "score_used_minerals_upgrade",
                                "score_used_vespene_none","score_used_vespene_army", "score_used_vespene_economy", "score_used_vespene_technology", "score_used_vespene_upgrade",
                                "score_total_used_minerals_none", "score_total_used_minerals_army", "score_total_used_minerals_economy", "score_total_used_minerals_technology", "score_total_used_minerals_upgrade",
                                "score_total_used_vespene_none", "score_total_used_vespene_army", "score_total_used_vespene_economy", "score_total_used_vespene_technology", "score_total_used_vespene_upgrade",
                                # Score by vital
                                "score_by_vital_total_damage_dealt_life", "score_by_vital_total_damage_dealt_shields", "score_by_vital_total_damage_dealt_energy",
                                "score_by_vital_total_damage_taken_life", "score_by_vital_total_damage_taken_shields", "score_by_vital_total_damage_taken_energy",
                                "score_by_vital_total_healed_life", "score_by_vital_total_healed_shields", "score_by_vital_total_healed_energy",
                                # Neutral units
                                "num_minerals", "num_geysers",
                                # Enemy info
                                "enemy_num_buildings", "enemy_total_building_health", "enemy_num_workers", "enemy_num_army_units", "enemy_total_army_health",
                            ])

class DQNAgent(BaseAgent):
    def __init__(self, main_network: DQNNetwork, buffer: ExperienceReplayBuffer,
                 hyperparams: DQNAgentParams,
                 target_network: DQNNetwork = None,
                 is_training: bool = True,
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

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.main_network = main_network
        self.target_network = target_network or deepcopy(main_network)
        self.buffer = buffer
        self.hyperparams = hyperparams
        self.initial_epsilon = hyperparams.epsilon
        self._is_training = is_training
        self.initialize()
#         torch.nn.MSELoss()
        self.loss = self.hyperparams.loss or torch.nn.HuberLoss()
        # Placeholders
        # self.state = None
        self._action_to_idx = {idx: action for idx, action in enumerate(self.agent_actions)}
        self._idx_to_action = {action: idx for idx, action in enumerate(self.agent_actions)}
        self._num_actions = len(self.agent_actions)

        self.__prev_state = None
        self.__prev_reward = None
        self.__prev_action = None
        self.__prev_action_args = None


    def initialize(self):
        # This should be exactly the same as self.steps (implemented by pysc2 base agent)
        self.__step_count = 0
        # Loss on each update
        self.__update_loss = []
        # Rewards at the end of each episode
        self.__training_rewards = []
        # Eps value at the end of each episode
        self.__episode_eps = []
        # Mean rewards at the end of each episode
        self.__mean_rewards = []
        # Rolling average of the rewards of the last 10 episodes
        self.__mean_rewards_10 = []
        # Loss on each episode
        self.__episode_losses = []
        # Steps performed on each episode
        self.__episode_steps = []
        # Average number of steps per episode
        self.__mean_steps = []
        # Highest reward for any episode
        self.__max_episode_rewards = None
        self.__episode_reward = 0

    def reset(self, **kwargs):
        """Initialize the agent."""
        super().reset(**kwargs)

        # This should be exactly the same as self.steps (implemented by pysc2 base agent)
        self.__step_count = 0
        # Last observation
        self.__prev_state = None
        self.__prev_reward = None
        self.__prev_action = None
        self.__prev_action_args = None
        self.__current_state = None
        self.__episode_reward = 0

    def _convert_obs_to_state(self, obs: TimeStep) -> State:
        building_state = self._get_buildings_state(obs)
        worker_state = self._get_workers_state(obs)
        army_state = self._get_army_state(obs)
        resources_state = self._get_resources_state(obs)
        scores_state = self._get_scores_state(obs)
        neutral_units_state = self._get_neutral_units_state(obs)
        enemy_state = self._get_enemy_state(obs)
        # Enemy

        return State(
			**building_state,
			**worker_state,
			**army_state,
			**resources_state,
            **scores_state,
            **neutral_units_state,
            **enemy_state
        )

    def select_action(self, obs: TimeStep) -> Tuple[AllActions, Dict[str, Any]]:
        available_actions = self.available_actions(obs)
        action = self.main_network.get_action(self.__current_state, self.hyperparams.epsilon, available_actions)
        action_args = self._get_action_args(obs=obs, action=action)

        return action, action_args

    def step(self, obs: TimeStep):
        action_call = super().step(obs)

        num_steps += 1
        self.__episode_reward += obs.reward

        if is_training:
            episode_losses = self.update_main_network(episode_losses)
            self.synchronize_target_network()

        if is_training:
            # Add the episode rewards to the training rewards
            self.training_rewards.append(episode_reward)
            # Register the epsilon used in the last episode played
            self.episode_eps.append(self.hyperparams.epsilon)
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
                + f"Epsilon {self.hyperparams.epsilon:.4f} :: "
                + f"Maxim {self.max_episode_rewards:.2f} :: "
                + f"Steps: {num_steps}\t\t\t\t", end="")


        return action_call

    def pre_step(self, obs: TimeStep):
        self.__current_state = self._convert_obs_to_state(obs)
        reward = obs.reward
        self.__episode_reward += reward
        self.__step_count += 1

        if not obs.first():
            # do updates
            done = obs.last()
            self.buffer.append(self.__prev_state, self.__prev_action, self.__prev_action_args, reward, done, self.__current_state)

            if self._is_training:
                self.__episode_losses = self.update_main_network(self.__episode_losses)
                self.synchronize_target_network()
                # HERE

    def post_step(self, obs: TimeStep, action: AllActions, action_args: Dict[str, Any]):
        self.__prev_state = self.__current_state
        self.__prev_action = action
        self.__prev_action_args = action_args


    # def post_step(self, obs: TimeStep, action: AllActions):
    #     self.__prev_action = action

    # def step(self, obs: TimeStep) -> AllActions:
    #     super().step(obs)
    #     if obs.first():
    #         self._setup_positions(obs)

    #     self.update_supply_depot_positions(obs)
    #     self.update_command_center_positions(obs)
    #     self.update_barracks_positions(obs)

    #     obs = self.preprocess_observation(obs)
    #     action, action_args = self.select_action(obs)

    #     self.logger.info(f"Performing action {action.name} with args: {action_args}")
    #     action = self._action_to_game[action]

    #     if action_args is not None:
    #         return action(**action_args)

    #     return action()




    # def take_action(self, action: int) -> Tuple[bool, float]:
    #     """Take an action in the environment.

    #     If the episode is finished after taking the action, the environment is reset.

    #     Args:
    #         action (int): Action to take

    #     Returns:
    #         Tuple[bool, float]: A bool indicating if the episode is finished, and a float with the reward of the step.
    #     """

    #     new_state, reward, done, truncated, _ = self.env.step(action)
    #     done = done or truncated

    #     if not done:
    #         self.buffer.append(self.state, action, reward, done, new_state)
    #         self.state = new_state

    #     if done:
    #         self.state = self.env.reset()[0]

    #     return done, reward

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
        eps = self.hyperparams.epsilon if is_training else 0
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
            self.episode_eps.append(self.hyperparams.epsilon)
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
                + f"Epsilon {self.hyperparams.epsilon:.4f} :: "
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
        self.hyperparams.epsilon = self.initial_epsilon

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

            self.hyperparams.epsilon = max(self.hyperparams.epsilon * self.hyperparams.epsilon_decay, self.hyperparams.min_epsilon)

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

        expected_qvals = self.hyperparams.gamma * qvals_next + rewards_vals

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
        if (self.steps % self.hyperparams.main_network_update_frequency != 0) and not force_update:
            return episode_losses

        self.main_network.optimizer.zero_grad()  # eliminem qualsevol gradient passat
        batch = self.buffer.sample_batch(batch_size=self.hyperparams.batch_size) # seleccionem un conjunt del buffer

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
        if (self.steps % self.hyperparams.target_network_sync_frequency != 0) and not force_update:
            return

        if self.hyperparams.target_sync_mode == "soft":
            for target_var, var in zip(self.target_network.parameters(), self.main_network.parameters()):
                    target_var.data.copy_((1. - self.hyperparams.update_tau) * target_var.data + (self.hyperparams.update_tau) * var.data)
        else:
            self.target_network.load_state_dict(self.main_network.state_dict())

    def _get_buildings_state(self, obs):
        def _num_complete(buildings):
            return len(list(filter(self.is_complete, buildings)))
        # info about command centers
        command_centers = self.get_self_units(obs, unit_types=units.Terran.CommandCenter)
        num_command_centers = len(command_centers)
        num_completed_command_centers = _num_complete(command_centers)
        command_centers_state = dict(
            num_command_centers=num_command_centers,
            num_completed_command_centers=num_completed_command_centers
        )
        for idx, cc in command_centers:
            if idx > 3:
                self.logger.warning(f"Observation space is only ready to consider queues and num workers for 4 command centers, but there are {num_command_centers}.")
                break
            command_centers_state[f"command_center_{idx}_order_length"] = cc.order_length
            command_centers_state[f"command_center_{idx}_num_workers"] = cc.assigned_harvesters

        # Buildings
        supply_depots = self.get_self_units(obs, unit_types=units.Terran.SupplyDepot)
        num_supply_depots = len(supply_depots)
        refineries = self.get_self_units(obs, unit_types=units.Terran.Refinery)
        num_refineries = len(refineries)
        barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks)
        num_barracks = len(len(barracks))
        other_building_types = [
            bt for bt in Constants.BUILDING_UNIT_TYPES
            if bt not in [units.Terran.CommandCenter, units.Terran.SupplyDepot, units.Terran.Refinery, units.Terran.Barracks]
        ]
        other_buildings = self.get_self_units(obs, unit_types=other_building_types)

        other_buildings = [b for b in other_buildings if b not in []]
        num_other_buildings = len(other_buildings)
        buildings_state = dict(
			num_refineries=num_refineries,
			num_completed_refineries=_num_complete(refineries),
			# Supply Depots
			num_supply_depots=num_supply_depots,
			num_completed_supply_depots=_num_complete(supply_depots),
			# Barracks
			num_barracks=num_barracks,
			num_completed_barracks=_num_complete(barracks),
            num_other_buildings=num_other_buildings
        )

        return {
            **command_centers_state,
            **buildings_state
        }

    def _get_workers_state(self, obs: TimeStep) -> Dict[str, int|float]:
        workers = self.get_workers(obs)
        num_harvesters = [w for w in workers if w.order_id_0 in self.HARVEST_ACTIONS]
        refineries = self.get_self_units(obs, unit_types=units.Terran.Refinery)
        num_gas_harvesters = sum(map(lambda r: r.assigned_harvesters, refineries))
        num_mineral_harvesters = num_harvesters - num_gas_harvesters
        num_workers = len(workers)
        num_idle_workers = len([w for w in workers if self.is_idle(w)])
        pct_idle_workers = num_idle_workers / num_workers
        pct_mineral_harvesters = num_mineral_harvesters / num_workers
        pct_gas_harvesters = num_gas_harvesters / num_workers

        # TODO more stats on N workers (e.g. distance to command centers, distance to minerals, to geysers...)
        return dict(
            num_workers=num_workers,
			num_idle_workers=len([w for w in workers if self.is_idle(w)]),
            pct_idle_workers=pct_idle_workers,
            num_mineral_harvesters=num_mineral_harvesters,
            pct_mineral_harvesters=pct_mineral_harvesters,
            num_gas_harvesters=num_gas_harvesters,
            pct_gas_harvesters=pct_gas_harvesters
        )

    def _get_army_state(self, obs: TimeStep) -> Dict[str, int|float]:
        marines = self.get_self_units(obs, unit_types=units.Terran.Marine)
        barracks = self.get_self_units(obs, unit_types=units.Terran.Barracks)
        num_marines_in_queue = sum(map(lambda b: b.order_length, barracks))
        other_army_units_types = [ut for ut in Constants.ARMY_UNIT_TYPES if ut not in units.Terran.Marine]
        other_army_units = self.get_self_units(obs, unit_types=other_army_units_types)
        return dict(
            num_marines=len(marines),
            num_marines_in_queue=num_marines_in_queue,
            num_other_army_units=len(other_army_units)
        )

    def _get_resources_state(self, obs: TimeStep) -> Dict[str, int|float]:
        return dict(
            free_supply=self.get_free_supply(obs),
            minerals=obs.observation.player.minerals,
            gas=obs.observation.player.vespene,
        )

    def _get_scores_state(self, obs: TimeStep)     -> Dict[str, int|float]:
        return {
            "score_cumulative": obs.observation.score_cumlative._index_names,
            "score_cumulative_score": obs.observation.score_cumulative.score,
            "score_cumulative_idle_production_time": obs.observation.score_cumulative.idle_production_time,
            "score_cumulative_idle_worker_time": obs.observation.score_cumulative.idle_worker_time,
            "score_cumulative_total_value_units": obs.observation.score_cumulative.total_value_units,
            "score_cumulative_total_value_structures": obs.observation.score_cumulative.total_value_structures,
            "score_cumulative_killed_value_units": obs.observation.score_cumulative.killed_value_units,
            "score_cumulative_killed_value_structures": obs.observation.score_cumulative.killed_value_structures,
            "score_cumulative_collected_minerals": obs.observation.score_cumulative.collected_minerals,
            "score_cumulative_collected_vespene": obs.observation.score_cumulative.collected_vespene,
            "score_cumulative_collection_rate_minerals": obs.observation.score_cumulative.collection_rate_minerals,
            "score_cumulative_collection_rate_vespene": obs.observation.score_cumulative.collection_rate_vespene,
            "score_cumulative_spent_minerals": obs.observation.score_cumulative.spent_minerals,
            "score_cumulative_spent_vespene": obs.observation.score_cumulative.spent_vespene,
            # Supply (food) scores
            "score_food_used_none": obs.observation.score_by_category.food_used.none,
            "score_food_used_army": obs.observation.score_by_category.food_used.army,
            "score_food_used_economy": obs.observation.score_by_category.food_used.economy,
            "score_food_used_technology": obs.observation.score_by_category.food_used.technology,
            "score_food_used_upgrade": obs.observation.score_by_category.food_used.upgrade,
            "score_by_vital": obs.observation.score_by_vital,
            # Killed minerals and vespene
            "score_killed_minerals_none": obs.observation.score_by_category.killed_minerals.none,
            "score_killed_minerals_army": obs.observation.score_by_category.killed_minerals.army,
            "score_killed_minerals_economy": obs.observation.score_by_category.killed_minerals.economy,
            "score_killed_minerals_technology": obs.observation.score_by_category.killed_minerals.technology,
            "score_killed_minerals_upgrade": obs.observation.score_by_category.killed_minerals.upgrade,
            "score_killed_minerals_none": obs.observation.score_by_category.killed_minerals.none,
            "score_killed_vespene_army": obs.observation.score_by_category.killed_vespene.army,
            "score_killed_vespene_economy": obs.observation.score_by_category.killed_vespene.economy,
            "score_killed_vespene_technology": obs.observation.score_by_category.killed_vespene.technology,
            "score_killed_vespene_upgrade": obs.observation.score_by_category.killed_vespene.upgrade,
            "score_killed_vespene_none": obs.observation.score_by_category.killed_vespene.none,
            # Lost minerals and vespene
            "score_lost_minerals_none": obs.observation.score_by_category.lost_minerals.none,
            "score_lost_minerals_army": obs.observation.score_by_category.lost_minerals.army,
            "score_lost_minerals_economy": obs.observation.score_by_category.lost_minerals.economy,
            "score_lost_minerals_technology": obs.observation.score_by_category.lost_minerals.technology,
            "score_lost_minerals_upgrade": obs.observation.score_by_category.lost_minerals.upgrade,
            "score_lost_minerals_none": obs.observation.score_by_category.lost_minerals.none,
            "score_lost_vespene_army": obs.observation.score_by_category.lost_vespene.army,
            "score_lost_vespene_economy": obs.observation.score_by_category.lost_vespene.economy,
            "score_lost_vespene_technology": obs.observation.score_by_category.lost_vespene.technology,
            "score_lost_vespene_upgrade": obs.observation.score_by_category.lost_vespene.upgrade,
            "score_lost_vespene_none": obs.observation.score_by_category.lost_vespene.none,
            # Friendly fire minerals and vespene
            "score_friendly_fire_minerals_none": obs.observation.score_by_category.friendly_fire_minerals.none,
            "score_friendly_fire_minerals_army": obs.observation.score_by_category.friendly_fire_minerals.army,
            "score_friendly_fire_minerals_economy": obs.observation.score_by_category.friendly_fire_minerals.economy,
            "score_friendly_fire_minerals_technology": obs.observation.score_by_category.friendly_fire_minerals.technology,
            "score_friendly_fire_minerals_upgrade": obs.observation.score_by_category.friendly_fire_minerals.upgrade,
            "score_friendly_fire_vespene_none": obs.observation.score_by_category.friendly_fire_vespene.none,
            "score_friendly_fire_vespene_army": obs.observation.score_by_category.friendly_fire_vespene.army,
            "score_friendly_fire_vespene_economy": obs.observation.score_by_category.friendly_fire_vespene.economy,
            "score_friendly_fire_vespene_technology": obs.observation.score_by_category.friendly_fire_vespene.technology,
            "score_friendly_fire_vespene_upgrade": obs.observation.score_by_category.friendly_fire_vespene.upgrade,
            # Used minerals and vespene
            "score_used_minerals_none": obs.observation.score_by_category.used_minerals.none,
            "score_used_minerals_army": obs.observation.score_by_category.used_minerals.army,
            "score_used_minerals_economy": obs.observation.score_by_category.used_minerals.economy,
            "score_used_minerals_technology": obs.observation.score_by_category.used_minerals.technology,
            "score_used_minerals_upgrade": obs.observation.score_by_category.used_minerals.upgrade,
            "score_used_vespene_none": obs.observation.score_by_category.usede_vespene.none,
            "score_used_vespene_army": obs.observation.score_by_category.usede_vespene.army,
            "score_used_vespene_economy": obs.observation.score_by_category.usede_vespene.economy,
            "score_used_vespene_technology": obs.observation.score_by_category.usede_vespene.technology,
            "score_used_vespene_upgrade": obs.observation.score_by_category.usede_vespene.upgrade,
            # Total used minerals and vespene
            "score_total_used_minerals_none": obs.observation.score_by_category.total_used_minerals.none,
            "score_total_used_minerals_army": obs.observation.score_by_category.total_used_minerals.army,
            "score_total_used_minerals_economy": obs.observation.score_by_category.total_used_minerals.economy,
            "score_total_used_minerals_technology": obs.observation.score_by_category.total_used_minerals.technology,
            "score_total_used_minerals_upgrade": obs.observation.score_by_category.total_used_minerals.upgrade,
            "score_total_used_vespene_none": obs.observation.score_by_category.total_usede_vespene.none,
            "score_total_used_vespene_army": obs.observation.score_by_category.total_usede_vespene.army,
            "score_total_used_vespene_economy": obs.observation.score_by_category.total_usede_vespene.economy,
            "score_total_used_vespene_technology": obs.observation.score_by_category.total_usede_vespene.technology,
            "score_total_used_vespene_upgrade": obs.observation.score_by_category.total_usede_vespene.upgrade,

            # Score by vital
            "score_by_vital_total_damage_dealt_life": obs.observation.score_by_vital.total_damage_dealt.life,
            "score_by_vital_total_damage_dealt_shields": obs.observation.score_by_vital.total_damage_dealt.shields,
            "score_by_vital_total_damage_dealt_energy": obs.observation.score_by_vital.total_damage_dealt.energy,
            "score_by_vital_total_damage_taken_life": obs.observation.score_by_vital.total_damage_taken.life,
            "score_by_vital_total_damage_taken_shields": obs.observation.score_by_vital.total_damage_taken.shields,
            "score_by_vital_total_damage_taken_energy": obs.observation.score_by_vital.total_damage_taken.energy,
            "score_by_vital_total_healed_life": obs.observation.score_by_vital.total_healed.life,
            "score_by_vital_total_healed_shields": obs.observation.score_by_vital.total_healed.shields,
            "score_by_vital_total_healed_energy": obs.observation.score_by_vital.total_healed.energy,
        }

    def _get_neutral_units_state(self, obs: TimeStep) -> Dict[str, int|float]:
        minerals = [unit.tag for unit in obs.observation.raw_units if Minerals.contains(unit.unit_type)]
        geysers = [unit.tag for unit in obs.observation.raw_units if Gas.contains(unit.unit_type)]
        return dict(
            num_minerals=len(minerals),
            num_geysers=len(geysers),
        )

    def _get_enemy_state(self, obs: TimeStep) -> Dict[str, int|float]:
        enemy_buildings = self.get_enemy_units(obs, unit_types=Constants.BUILDING_UNIT_TYPES)
        enemy_workers = self.get_enemy_units(obs, unit_types=Constants.WORKER_UNIT_TYPES)
        enemy_army = self.get_enemy_units(obs, unit_types=Constants.ARMY_UNIT_TYPES)

        return dict(
            enemy_num_buildings=len(enemy_buildings),
            enemy_total_building_health=sum(map(lambda b: b.health, enemy_buildings)),
            enemy_num_workers=len(enemy_workers),
            enemy_num_army_units = len(enemy_army),
            enemy_total_army_health=sum(map(lambda b: b.health, enemy_army)),
        )
