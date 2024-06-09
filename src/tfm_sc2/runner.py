import logging
import pickle
from pathlib import Path

import torch
from absl import app, flags
from codecarbon import OfflineEmissionsTracker
from pysc2.env import sc2_env
from pysc2.lib.remote_controller import ConnectError
from tfm_sc2.actions import (
    AllActions,
    ArmyAttackManagerActions,
    ArmyRecruitManagerActions,
    BaseManagerActions,
    GameManagerActions,
)
from tfm_sc2.agents.dqn_agent import DQNAgentParams, State
from tfm_sc2.agents.multi.army_attack_manager_dqn_agent import ArmyAttackManagerDQNAgent
from tfm_sc2.agents.multi.army_attack_manager_random_agent import (
    ArmyAttackManagerRandomAgent,
)
from tfm_sc2.agents.multi.army_recruit_manager_dqn_agent import (
    ArmyRecruitManagerDQNAgent,
)
from tfm_sc2.agents.multi.army_recruit_manager_random_agent import (
    ArmyRecruitManagerRandomAgent,
)
from tfm_sc2.agents.multi.base_manager_dqn_agent import BaseManagerDQNAgent
from tfm_sc2.agents.multi.base_manager_random_agent import BaseManagerRandomAgent
from tfm_sc2.agents.multi.game_manager_dqn_agent import GameManagerDQNAgent
from tfm_sc2.agents.multi.game_manager_random_agent import GameManagerRandomAgent
from tfm_sc2.agents.single.single_dqn_agent import SingleDQNAgent
from tfm_sc2.agents.single.single_random_agent import SingleRandomAgent
from tfm_sc2.agents.single.single_scripted_agent import SingleScriptedAgent
from tfm_sc2.networks.dqn_network import DQNNetwork
from tfm_sc2.networks.experience_replay_buffer import ExperienceReplayBuffer
from tfm_sc2.sc2_config import MAP_CONFIGS, SC2_CONFIG
from tfm_sc2.types import RewardMethod
from tfm_sc2.with_logger import WithLogger
from torch import optim

FLAGS = flags.FLAGS

class MainLogger:
    _logger = None
    @classmethod
    def get(cls) -> logging.Logger:
        if cls._logger is None:
            cls._logger = logging.getLogger("main_runner")
        return cls._logger

def setup_logging(log_file: str = None):
    if log_file  is not None:
        log_file: Path = Path(log_file)
        log_file.parent.mkdir(exist_ok=True, parents=True)
    WithLogger.init_logging(stream_level=logging.DEBUG, file_name=log_file)
    absl_logger = logging.getLogger("absl")
    absl_logger.setLevel(logging.WARNING)

def load_dqn_agent(cls, map_name, map_config, checkpoint_path: Path, exploit: bool, buffer: ExperienceReplayBuffer = None, **extra_agent_args):
    MainLogger.get().info(f"Loading agent from file {checkpoint_path}")
    agent = cls.load(checkpoint_path, map_name=map_name, map_config=map_config, buffer=buffer, **extra_agent_args)
    if exploit:
        agent.exploit()
    else:
        agent.train()

    return agent

def load_dqn(network_path):
    dqn = torch.load(network_path)
    return dqn

def create_dqn(cls = None):
    logger = MainLogger.get()
    if cls in [GameManagerDQNAgent, GameManagerRandomAgent]:
        num_actions = len(GameManagerActions)
        logger.info(f"Creating DQN with {num_actions} actions from GameManagerActions for agent {cls}")
    elif cls in [BaseManagerDQNAgent, BaseManagerRandomAgent]:
        num_actions = len(BaseManagerActions)
        logger.info(f"Creating DQN with {num_actions} actions from BaseManagerActions for agent {cls}")
    elif cls in [ArmyRecruitManagerDQNAgent, ArmyRecruitManagerRandomAgent]:
        num_actions = len(ArmyRecruitManagerActions)
        logger.info(f"Creating DQN with {num_actions} actions from ArmyRecruitManagerActions for agent {cls}")
    elif cls in [ArmyAttackManagerDQNAgent, ArmyAttackManagerRandomAgent]:
        num_actions = len(ArmyAttackManagerActions)
        logger.info(f"Creating DQN with {num_actions} actions from ArmyAttackManagerActions for agent {cls}")
    else:
        num_actions = len(AllActions)
        logger.info(f"Creating DQN with {num_actions} actions from AllActions for agent {cls}")

    match FLAGS.dqn_size:
        case "extra_small":
            model_layers = [64, 32]
            logger.info(f"Using extra small network ({model_layers}) for agent {cls}")
        case "small":
            model_layers = [128, 64, 32]
            logger.info(f"Using small network ({model_layers}) for agent {cls}")
        case "medium":
            model_layers = [128, 128, 64, 32]
            logger.info(f"Using medium network ({model_layers}) for agent {cls}")
        case "large":
            model_layers = [256, 128, 128, 64, 32]
            logger.info(f"Using large network ({model_layers}) for agent {cls}")
        case "extra_large":
            model_layers = [512, 256, 128, 128, 64, 32]
            logger.info(f"Using extra large network ({model_layers}) for agent {cls}")

    obs_input_shape = len(State._fields)
    learning_rate = FLAGS.lr
    lr_milestones = FLAGS.lr_milestones
    dqn = DQNNetwork(model_layers=model_layers, observation_space_shape=obs_input_shape, num_actions=num_actions, learning_rate=learning_rate, lr_milestones=lr_milestones)

    return dqn

def create_dqn_agent(cls, map_name, map_config, main_network: DQNNetwork, checkpoint_path: Path, log_name: str, exploit: bool, reward_method: RewardMethod, memory_size: int = 100000, burn_in: int = 10000, target_network: DQNNetwork = None, buffer: ExperienceReplayBuffer = None, **extra_agent_args):
    if buffer is None:
        buffer = ExperienceReplayBuffer(memory_size=memory_size, burn_in=burn_in)

    # Update the main net every 50 agent steps, and the target network at the end of the episode
    agent_params = DQNAgentParams(epsilon=0.9, epsilon_decay=FLAGS.epsilon_decay, min_epsilon=0.01, batch_size=512, gamma=0.99, main_network_update_frequency=50, target_network_sync_frequency=-1, target_sync_mode="soft", update_tau=0.1)
    # agent_params = DQNAgentParams(epsilon=0.9, epsilon_decay=0.99, min_epsilon=0.01, batch_size=512, gamma=0.99, main_network_update_frequency=-1, target_network_sync_frequency=-1, target_sync_mode="soft", update_tau=0.5)
    agent = cls(map_name=map_name, map_config=map_config, main_network=main_network, target_network=target_network, buffer=buffer, hyperparams=agent_params, checkpoint_path=checkpoint_path, log_name=log_name, reward_method=reward_method, **extra_agent_args)

    if exploit:
        agent.exploit()
    else:
        agent.train()

    return agent

def load_or_create_dqn_agent(cls, map_name, map_config, load_agent: bool, checkpoint_path: Path, exploit: bool, log_name: str, load_networks_only: bool, reward_method: RewardMethod, memory_size: int = 100000, burn_in: int = 10000, buffer: ExperienceReplayBuffer = None, **extra_agent_args):
    if load_networks_only:
        if buffer is None:
            buffer = ExperienceReplayBuffer(memory_size=memory_size, burn_in=burn_in)
        main_network = load_dqn(checkpoint_path / SingleDQNAgent._MAIN_NETWORK_FILE)
        target_network = load_dqn(checkpoint_path / SingleDQNAgent._TARGET_NETWORK_FILE)
        agent = create_dqn_agent(cls=cls, map_name=map_name, map_config=map_config, checkpoint_path=checkpoint_path, main_network=main_network, target_network=target_network, buffer=buffer, log_name=log_name, exploit=exploit, reward_method=reward_method, **extra_agent_args)
    elif not load_agent:
        main_network = create_dqn(cls)
        if buffer is None:
            buffer = ExperienceReplayBuffer(memory_size=memory_size, burn_in=burn_in)
        agent = create_dqn_agent(cls=cls, map_name=map_name, map_config=map_config, checkpoint_path=checkpoint_path, main_network=main_network, buffer=buffer, log_name=log_name, exploit=exploit, reward_method=reward_method, **extra_agent_args)
    else:
        # HEre we allow buffer to be None. In that case, the agent's buffer is loaded, otherwise it will be replaced
        agent = load_dqn_agent(cls=cls, map_name=map_name, map_config=map_config, checkpoint_path=checkpoint_path, exploit=exploit, buffer=buffer,  **extra_agent_args)
        if FLAGS.reset_epsilon:
            agent.epsilon = agent.initial_epsilon

    return agent

def load_random_agent(cls, map_name, map_config, checkpoint_path: Path, buffer: ExperienceReplayBuffer = None, **extra_agent_args):
    MainLogger.get().info(f"Loading agent from file {checkpoint_path}")
    agent = cls.load(checkpoint_path, map_name=map_name, map_config=map_config, buffer=buffer, **extra_agent_args)

    return agent

def create_random_agent(cls, map_name, map_config, checkpoint_path: Path, log_name: str, reward_method: RewardMethod, memory_size: int = 100000, burn_in: int = 10000, buffer: ExperienceReplayBuffer = None, **extra_agent_args):
    if buffer is None:
        buffer = ExperienceReplayBuffer(memory_size=memory_size, burn_in=burn_in)
    agent = cls(map_name=map_name, map_config=map_config, log_name=log_name, buffer=buffer, checkpoint_path=checkpoint_path, reward_method=reward_method, **extra_agent_args)

    return agent


def load_or_create_random_agent(cls, map_name, map_config, load_agent: bool, checkpoint_path: Path, log_name: str, reward_method: RewardMethod, memory_size: int = 100000, burn_in: int = 10000, buffer: ExperienceReplayBuffer = None, **extra_agent_args):
    if load_agent:
        agent = load_random_agent(cls=cls, map_name=map_name, map_config=map_config, checkpoint_path=checkpoint_path, buffer=buffer,  **extra_agent_args)
    else:
        if buffer is None:
            buffer = ExperienceReplayBuffer(memory_size=memory_size, burn_in=burn_in)
        agent = create_random_agent(cls=cls, map_name=map_name, map_config=map_config, buffer=buffer, checkpoint_path=checkpoint_path, log_name=log_name, reward_method=reward_method, **extra_agent_args)

    return agent

def main(unused_argv):
    FLAGS = flags.FLAGS
    setup_logging(FLAGS.log_file)
    logger = MainLogger.get()
    SC2_CONFIG["visualize"] = not FLAGS.no_visualize

    map_name = FLAGS.map_name
    if map_name not in MAP_CONFIGS:
        raise RuntimeError(f"No config for map {map_name}")
    map_config = MAP_CONFIGS[map_name]
    model_id = FLAGS.model_id or FLAGS.agent_key
    checkpoint_path = Path(FLAGS.models_path) / model_id
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    emissions_filename = "emissions.csv"
    emissions_file = checkpoint_path / emissions_filename
    emissions_idx = 0
    while emissions_file.exists():
        emissions_idx += 1
        if emissions_idx >= 100:
            raise RuntimeError(f"There are already 100 emission files under {checkpoint_path}, please clean them up or move them to continue")
        emissions_filename = f"emissions_{emissions_idx:02d}.csv"
        emissions_file = checkpoint_path / emissions_filename
    save_path = checkpoint_path
    # checkpoint_path: Path = None
    agent_file = checkpoint_path / "agent.pkl"
    load_agent = agent_file.exists()
    load_networks_only = FLAGS.load_networks_only
    if load_agent:
        logger.info(f"Agent will be loaded from file: {agent_file}")
    else:
        logger.info(f"A new agent will be created")
    save_agent = True
    exploit = FLAGS.exploit
    # We will still save the stats when exploiting, but in a subfolder
    if exploit:
        save_path = checkpoint_path / "exploit"
        save_path.mkdir(exist_ok=True, parents=True)

    reward_method_str = FLAGS.reward_method

    match reward_method_str:
        case "score":
            reward_method = RewardMethod.SCORE
        case "adjusted_reward":
            reward_method = RewardMethod.ADJUSTED_REWARD
        case "reward":
            reward_method = RewardMethod.REWARD
        case _:
            logger.warning(f"Unknown reward_method flag {reward_method_str}, falling back to REWARD")
            reward_method = RewardMethod.REWARD

    logger.info(f"Map name: {map_name}")
    logger.info(f"Map available actions: {map_config['available_actions']}")

    other_agents = []
    buffer_file = FLAGS.buffer_file
    memory_size = FLAGS.memory_size
    burn_in = FLAGS.burn_in

    def _create_random_enemy_gm():
        bm_buffer = ExperienceReplayBuffer(memory_size=memory_size, burn_in=burn_in)
        base_manager = create_random_agent(cls=BaseManagerRandomAgent, map_name=map_name, map_config=map_config, checkpoint_path=None, buffer=bm_buffer, log_name=f"Enemy SubAgent {len(other_agents) + 1} - BaseManager", reward_method=reward_method)
        base_manager.logger.setLevel(logging.WARNING)
        arm_buffer = ExperienceReplayBuffer(memory_size=memory_size, burn_in=burn_in)
        army_recruit_manager = create_random_agent(cls=ArmyRecruitManagerRandomAgent, map_name=map_name, map_config=map_config, checkpoint_path=None, buffer=arm_buffer, log_name=f"Enemy SubAgent {len(other_agents) + 1} - ArmyRecruitManager", reward_method=reward_method)
        army_recruit_manager.logger.setLevel(logging.WARNING)
        aam_buffer = ExperienceReplayBuffer(memory_size=memory_size, burn_in=burn_in)
        army_attack_manager = create_random_agent(cls=ArmyAttackManagerRandomAgent, map_name=map_name, map_config=map_config, checkpoint_path=None, buffer=aam_buffer, log_name=f"Enemy SubAgent {len(other_agents) + 1} - ArmyAttackManager", reward_method=reward_method)
        army_attack_manager.logger.setLevel(logging.WARNING)
        extra_agent_args = dict(
            base_manager=base_manager,
            army_recruit_manager=army_recruit_manager,
            army_attack_manager=army_attack_manager
        )
        gm_buffer = ExperienceReplayBuffer(memory_size=memory_size, burn_in=burn_in)
        game_manager = create_random_agent(cls=GameManagerRandomAgent, map_name=map_name, map_config=map_config, checkpoint_path=None, buffer=gm_buffer, log_name=f"Enemy {len(other_agents) + 1} - GameManager", reward_method=reward_method, **extra_agent_args)
        game_manager.logger.setLevel(logging.WARNING)
        return game_manager

    if (len(map_config["players"]) > 1) and FLAGS.agent2_path is None:
        for other_agent_type in map_config["players"][1:]:
            if isinstance(other_agent_type, sc2_env.Agent):
                # if FLAGS.agent_key.startswith("single"):
                if not FLAGS.use_scripted_enemy:
                    logger.info(f"Adding random single agent as opponent #{len(other_agents) + 1}#")
                    enemy_agent = SingleRandomAgent(map_name=map_name, map_config=map_config, log_name=f"Random Agent {len(other_agents) + 1}", log_level=logging.ERROR, reward_method=reward_method)
                    enemy_agent.logger.setLevel(logging.WARNING)
                else:
                    logger.info(f"Adding scripted single agent as opponent #{len(other_agents) + 1}#")
                    enemy_agent = SingleScriptedAgent(map_name=map_name, map_config=map_config, log_name=f"Scripted Agent {len(other_agents) + 1}", log_level=logging.ERROR, reward_method=reward_method)
                    enemy_agent.logger.setLevel(logging.WARNING)
                other_agents.append(enemy_agent)
                # elif FLAGS.agent_key.startswith("multi"):
                #     logger.info(f"Adding random multi agent as opponent #{len(other_agents) + 1}#")
                #     other_agents.append(_create_random_enemy_gm())
    elif (len(map_config["players"]) > 1):
        logger.info("Loading SingleRandom agent as enemy")
        enemy_agent = load_or_create_dqn_agent(SingleDQNAgent, exploit=exploit, checkpoint_path=FLAGS.agent2_path, load_agent=True, map_name=map_name, map_config=map_config, reward_method=reward_method, log_name="Enemy Agent - SingleDQN", load_networks_only=False)
        other_agents.append(enemy_agent)

    if buffer_file is None:
        logger.info(f"Creating new buffer with memory size = {memory_size} and burn-in = {burn_in}")
        buffer = ExperienceReplayBuffer(memory_size=memory_size, burn_in=burn_in)
    else:
        logger.info(f"Using buffer from file {buffer_file}")
        with open(buffer_file, mode="rb") as f:
            buffer = pickle.load(f)

    base_args = dict(map_name=map_name, map_config=map_config, reward_method=reward_method)
    common_args = dict(buffer=buffer, load_agent=load_agent,  **base_args)
    dqn_agent_args = dict(exploit=exploit, load_networks_only=load_networks_only, checkpoint_path=checkpoint_path, **common_args)
    subagent_args = dict(buffer=None, exploit=True, load_networks_only=False, **base_args)
    match FLAGS.agent_key:
        case "single.random":
            agent = load_or_create_random_agent(cls=SingleRandomAgent, **common_args, log_name="Main Agent - Random")
        case "single.scripted":
            agent = load_or_create_random_agent(cls=SingleScriptedAgent, **common_args, log_name="Main Agent - Scripted")
        case "single.dqn":
            agent = load_or_create_dqn_agent(SingleDQNAgent, **dqn_agent_args, log_name="Main Agent - SingleDQN")
        case "multi.random.base_manager":
            agent = load_or_create_random_agent(BaseManagerRandomAgent, **common_args, log_name="Main Agent - Random BaseManager")
        case "multi.random.army_recruit_manager":
            agent = load_or_create_random_agent(ArmyRecruitManagerRandomAgent, **common_args, log_name="Main Agent - Random ArmyRecruitManager")
        case "multi.random.army_attack_manager":
            agent = load_or_create_random_agent(ArmyAttackManagerRandomAgent, **common_args, log_name="Main Agent - Random ArmyAttackManager")
        case "multi.random.game_manager":
            bm_path = checkpoint_path / "base_manager"
            arm_path = checkpoint_path / "army_recruit_manager"
            aam_path = checkpoint_path / "army_attack_manager"
            base_manager = load_or_create_random_agent(BaseManagerRandomAgent, **common_args, checkpoint_path=bm_path, log_name="SubAgent - Random BaseManager")
            base_manager.logger.setLevel(logging.WARNING)
            army_recruit_manager = load_or_create_random_agent(ArmyRecruitManagerRandomAgent, **common_args, checkpoint_path=arm_path, log_name="SubAgent - Random ArmyRecruitManager")
            army_recruit_manager.logger.setLevel(logging.WARNING)
            army_attack_manager = load_or_create_random_agent(ArmyAttackManagerRandomAgent, **common_args, checkpoint_path=aam_path, log_name="SubAgent - Random ArmyAttackManager")
            army_attack_manager.logger.setLevel(logging.WARNING)
            extra_agent_args = dict(
                base_manager=base_manager,
                army_recruit_manager=army_recruit_manager,
                army_attack_manager=army_attack_manager
            )
            agent = load_or_create_random_agent(GameManagerRandomAgent, **common_args, log_name="Main Agent - Random GameManager", **extra_agent_args)
        case "multi.dqn.base_manager":
            agent = load_or_create_dqn_agent(BaseManagerDQNAgent, **dqn_agent_args, log_name="Main Agent - Base Manager")
        case "multi.dqn.army_recruit_manager":
            agent = load_or_create_dqn_agent(ArmyRecruitManagerDQNAgent, **dqn_agent_args, log_name="Main Agent - Army Manager")
        case "multi.dqn.army_attack_manager":
            agent = load_or_create_dqn_agent(ArmyAttackManagerDQNAgent, **dqn_agent_args, log_name="Main Agent - Attack Manager")
        case "multi.dqn.game_manager" if not FLAGS.use_random_subagents:
            bm_path = checkpoint_path / "base_manager"
            bm_agent_file = bm_path / "agent.pkl"
            arm_path = checkpoint_path / "army_recruit_manager"
            arm_agent_file = arm_path / "agent.pkl"
            aam_path = checkpoint_path / "army_attack_manager"
            aam_agent_file = aam_path / "agent.pkl"
            # assert bm_agent_file.exists(), f"The agent file for the base manager doesn't exist '{bm_agent_file}'"
            # assert arm_agent_file.exists(), f"The agent file for the army recruit manager doesn't exist '{arm_agent_file}'"
            # assert aam_agent_file.exists(), f"The agent file for the attack manager doesn't exist '{arm_agent_file}'"
            logger.info("Loading base manager")
            base_manager = load_or_create_dqn_agent(BaseManagerDQNAgent, **subagent_args, load_agent=bm_agent_file.exists(), checkpoint_path=bm_path, log_name="Sub Agent - Base Manager")
            logger.info("Loading army recruit manager")
            army_recruit_manager = load_or_create_dqn_agent(ArmyRecruitManagerDQNAgent, **subagent_args, load_agent=arm_agent_file.exists(), checkpoint_path=arm_path, log_name="Sub Agent - Army Manager")
            logger.info("Loading attack manager")
            army_attack_manager = load_or_create_dqn_agent(ArmyAttackManagerDQNAgent, **subagent_args, load_agent=aam_agent_file.exists(), checkpoint_path=aam_path, log_name="Sub Agent - Attack Manager")
            extra_agent_args = dict(
                base_manager=base_manager,
                army_recruit_manager=army_recruit_manager,
                army_attack_manager=army_attack_manager
            )

            agent = load_or_create_dqn_agent(GameManagerDQNAgent, **dqn_agent_args, log_name="Main Agent - Game Manager", **extra_agent_args)
        case "multi.dqn.game_manager" if FLAGS.use_random_subagents:
            bm_path = checkpoint_path / "base_manager"
            arm_path = checkpoint_path / "army_recruit_manager"
            aam_path = checkpoint_path / "army_attack_manager"
            base_manager = load_or_create_random_agent(BaseManagerRandomAgent, **common_args, checkpoint_path=bm_path, log_name="SubAgent - Random BaseManager")
            base_manager.logger.setLevel(logging.WARNING)
            army_recruit_manager = load_or_create_random_agent(ArmyRecruitManagerRandomAgent, **common_args, checkpoint_path=arm_path, log_name="SubAgent - Random ArmyRecruitManager")
            army_recruit_manager.logger.setLevel(logging.WARNING)
            army_attack_manager = load_or_create_random_agent(ArmyAttackManagerRandomAgent, **common_args, checkpoint_path=aam_path, log_name="SubAgent - Random ArmyAttackManager")
            army_attack_manager.logger.setLevel(logging.WARNING)
            extra_agent_args = dict(
                base_manager=base_manager,
                army_recruit_manager=army_recruit_manager,
                army_attack_manager=army_attack_manager
            )

            agent = load_or_create_dqn_agent(GameManagerDQNAgent, **dqn_agent_args, log_name="Main Agent - Game Manager", **extra_agent_args)
        case _:
            raise RuntimeError(f"Unknown agent key {FLAGS.agent_key}")
    try:
        if FLAGS.export_stats_only:
            agent.save_stats(save_path)
            return
        finished_episodes = 0
        # We set measure_power_secs to a very high value because we want to flush emissions as we want
        tracker = OfflineEmissionsTracker(country_iso_code="ESP", experiment_id=f"global_{FLAGS.model_id}_{map_name}", measure_power_secs=3600, log_level=logging.WARNING, output_dir=str(checkpoint_path), output_file=emissions_filename)
        agent.set_tracker(tracker)
        max_episode_failures = 5
        current_episode_failures = 0
        tracker.start()
        if not exploit and hasattr(agent, "memory_replay_ready"):
            is_burnin = not agent.memory_replay_ready
            logger.info(f"Agent has a memory replay buffer. Requires burnin: {is_burnin}")
            burnin_episodes = 0
            while is_burnin and (burnin_episodes < FLAGS.max_burnin_episodes):
                try:
                    with sc2_env.SC2Env(
                        map_name=map_name,
                        players=map_config["players"],
                        **SC2_CONFIG) as env:

                        agent.setup(env.observation_spec(), env.action_spec())
                        for a in other_agents:
                            a.setup(env.observation_spec(), env.action_spec())

                        timesteps = env.reset()
                        agent.reset()
                        for a in other_agents:
                            a.reset()
                        episode_ended = timesteps[0].last()
                        while not episode_ended:
                            step_actions = [a.step(timestep) for a, timestep in zip([agent, *other_agents], timesteps)]
                            # step_actions = [agent.step(timesteps[0])]
                            timesteps = env.step(step_actions)
                            episode_ended = timesteps[0].last()
                            if episode_ended:
                                # Perform one last step to process rewards etc
                                last_step_actions = [a.step(timestep) for a, timestep in zip([agent, *other_agents], timesteps)]
                        current_episode_failures = 0
                        is_burnin = not agent.memory_replay_ready
                except ConnectError as error:
                    logger.warning("Couldn't connect to SC2 environment, trying to restart the episode again")
                    logger.warning(error)
                    burnin_episodes -= 1
                    current_episode_failures += 1
                    if current_episode_failures >= max_episode_failures:
                        logger.error(f"Reached max number of allowed episode failures, stopping run")
                        break
                finally:
                    burnin_episodes += 1
                    logger.info(f"Burnin progress: {100 * agent._buffer.burn_in_capacity:.2f}%")
            logger.info(f"Finished burnin after {burnin_episodes} episodes")

        num_wins = 0
        num_wins_enemy = 0
        num_games = 0
        while finished_episodes < FLAGS.num_episodes:
            already_saved = False
            try:
                with sc2_env.SC2Env(
                    map_name=map_name,
                    players=map_config["players"],
                    **SC2_CONFIG) as env:

                    agent.setup(env.observation_spec(), env.action_spec())
                    for a in other_agents:
                        a.setup(env.observation_spec(), env.action_spec())

                    timesteps = env.reset()
                    agent.reset()
                    for a in other_agents:
                        a.reset()
                    episode_ended = timesteps[0].last()
                    while not episode_ended:
                        step_actions = [a.step(timestep) for a, timestep in zip([agent, *other_agents], timesteps)]
                        # step_actions = [agent.step(timesteps[0])]
                        timesteps = env.step(step_actions)
                        episode_ended = timesteps[0].last()
                        if episode_ended:
                            # Perform one last step to process rewards etc
                            last_step_actions = [a.step(timestep) for a, timestep in zip([agent, *other_agents], timesteps)]

                            for idx, a in enumerate(other_agents):
                                logger.info(f"Total reward for enemy agent {idx + 1}: {a.reward}")
                                win_rate = 100 * (finished_episodes - a.reward) / finished_episodes
                                logger.info(f"Win rate for main agent: {win_rate:.2f}%")

                    current_episode_failures = 0
            except ConnectError as error:
                logger.error("Couldn't connect to SC2 environment, trying to restart the episode again")
                logger.error(error)
                finished_episodes -= 1
                current_episode_failures += 1
                if current_episode_failures >= max_episode_failures:
                    logger.error(f"Reached max number of allowed episode failures, stopping run")
                    break
            finally:
                finished_episodes += 1
                if save_agent and (finished_episodes % FLAGS.save_frequency_episodes) == 0:
                    logger.info(f"Saving agent after {finished_episodes} episodes")
                    agent.save(save_path)
                    already_saved = True
        if save_agent and not already_saved:
            logger.info(f"Saving final agent after {finished_episodes} episodes")
            total_emissions = tracker.stop()
            agent.save(save_path)
        else:
            total_emissions = tracker.stop()
            logger.info(f"Total emissions after {finished_episodes} episodes for agent {agent._log_name} (and {len(other_agents)} other agents): {total_emissions:.2f}")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    DEFAULT_MODELS_PATH = str(Path(__file__).parent.parent.parent / "models")

    agent_keys = [
        "single.random",
        "single.scripted",
        "single.dqn",
        "multi.random.base_manager",
        "multi.random.army_recruit_manager",
        "multi.random.army_attack_manager",
        "multi.random.game_manager",
        "multi.dqn.base_manager",
        "multi.dqn.army_recruit_manager",
        "multi.dqn.army_attack_manager",
        "multi.dqn.game_manager",
    ]
    map_keys = list(MAP_CONFIGS.keys())

    flags.DEFINE_enum("agent_key", "single.random", agent_keys, "Agent to use.")
    flags.DEFINE_enum("map_name", "Simple64", map_keys, "Map to use.")
    flags.DEFINE_enum("reward_method", default="reward", required=False, enum_values=["reward", "adjusted_reward", "score"], help="What to use to calculate rewards: reward = observation reward, adjusted_reward = observation reward with penalties for step, no-ops and invalid actions, or score deltas (i.e. score increase / decrease + penalty for invalid actions and no-ops).")
    flags.DEFINE_enum("dqn_size", "large", ["extra_small", "small", "medium", "large", "extra_large"], "Map to use.", required=False)
    flags.DEFINE_integer("num_episodes", 1, "Number of episodes to play.", lower_bound=1)

    flags.DEFINE_integer("memory_size", 100000, required=False, help="Total memory size for the buffer.", lower_bound=100)
    flags.DEFINE_integer("burn_in", 10000, required=False, help="Burn-in size for the buffer.", lower_bound=0)
    flags.DEFINE_float("epsilon_decay", 0.99, required=False, help="Epsilon decay for DQN agents.", lower_bound=0.01)
    #flags.DEFINE_integer("lr_milestone1", 0.99, required=False, help="Epsilon decay for DQN agents.", lower_bound=0.01)
    flags.DEFINE_float("lr", 1e-4, required=False, help="Learning rate for DQN agents.", upper_bound=1.)
    flags.DEFINE_list("lr_milestones", default=[], required=False, help="LR will be decayed (divided by 10) each time one of the episodes on this list is reached")
    flags.DEFINE_integer("max_burnin_episodes", 500, "Meximum number of episodes to allow to use for burning replay memories in.", lower_bound=0)
    flags.DEFINE_string("model_id", default=None, help="Determines the folder inside 'models_path' to save the model to", required=False)
    flags.DEFINE_string("models_path", help="Path where checkpoints are written to/loaded from", required=False, default=DEFAULT_MODELS_PATH)
    flags.DEFINE_string("agent2_path", help="Path to the enemy agent, if you want to use a SingleDQNAgent instead of a random agent as the enemy", required=False, default=None)
    flags.DEFINE_string("buffer_file", help="Path to a buffer to use instead of an empty buffer. Useful to skip burn-ins", required=False, default=None)
    flags.DEFINE_integer("save_frequency_episodes", default=1, help="We save the agent every X episodes.", lower_bound=1, required=False)
    flags.DEFINE_boolean("exploit", default=False, required=False, help="Use the agent in explotation mode, not for training.")
    flags.DEFINE_boolean("use_scripted_enemy", default=False, required=False, help="Use a scripted enemy instead of a random one.")
    flags.DEFINE_boolean("use_random_subagents", default=False, required=False, help="Use random sub-agents for the feudal agent.")
    flags.DEFINE_boolean("reset_epsilon", default=False, required=False, help="Reset epsilon to its default when loading an agent.")
    flags.DEFINE_boolean("save_agent", default=True, required=False, help="Whether to save the agent and/or its stats.")
    flags.DEFINE_boolean("random_mode", default=False, required=False, help="Tell the agent to run in random mode. Used mostly to ensure we collect experiences.")
    flags.DEFINE_boolean("export_stats_only", default=False, required=False, help="Set it to true if you only want to load the agent and export its stats.")
    flags.DEFINE_boolean("no_visualize", default=False, required=False, help="Set it to true if you don't want to visualize the games.")
    flags.DEFINE_boolean("load_networks_only", default=False, required=False, help="Provide this flag if you want to load DQN agents, but only load its networks (no buffers, params, etc). Might be useful for curriculum training.")
    flags.DEFINE_string("log_file", default=None, required=False, help="File to save detailed logs to")
    app.run(main)
