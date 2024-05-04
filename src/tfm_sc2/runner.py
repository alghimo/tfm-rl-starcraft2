import logging
from pathlib import Path

import torch
from absl import app, flags
from codecarbon import OfflineEmissionsTracker
from pysc2.env import sc2_env
from pysc2.lib.remote_controller import ConnectError
from tfm_sc2.actions import AllActions
from tfm_sc2.agents.dqn_agent import DQNAgentParams, State
from tfm_sc2.agents.multi.army_attack_manager_dqn_agent import ArmyAttackManagerDQNAgent
from tfm_sc2.agents.multi.army_recruit_manager_dqn_agent import (
    ArmyRecruitManagerDQNAgent,
)
from tfm_sc2.agents.multi.base_manager_dqn_agent import BaseManagerDQNAgent
from tfm_sc2.agents.multi.game_manager_dqn_agent import GameManagerDQNAgent
from tfm_sc2.agents.single.single_dqn_agent import SingleDQNAgent
from tfm_sc2.agents.single.single_random_agent import SingleRandomAgent
from tfm_sc2.networks.dqn_network import DQNNetwork
from tfm_sc2.networks.experience_replay_buffer import ExperienceReplayBuffer
from tfm_sc2.sc2_config import MAP_CONFIGS, SC2_CONFIG
from tfm_sc2.types import RewardMethod
from tfm_sc2.with_logger import WithLogger


class MainLogger:
    _logger = None
    @classmethod
    def get(cls) -> logging.Logger:
        if cls._logger is None:
            cls._logger = logging.getLogger("main_runner")
        return cls._logger

def setup_logging(log_file: str = None):
    log_file: Path = Path(log_file)
    log_file.parent.mkdir(exist_ok=True, parents=True)
    WithLogger.init_logging(file_name=log_file)
    absl_logger = logging.getLogger("absl")
    absl_logger.setLevel(logging.WARNING)

def load_dqn_agent(cls, map_name, map_config, checkpoint_path: Path, exploit: bool):
    MainLogger.get().info(f"Loading agent from file {checkpoint_path}")
    agent = cls.load(checkpoint_path, map_name=map_name, map_config=map_config)
    if exploit:
        agent.exploit()
    else:
        agent.train()

    return agent

def load_dqn(network_path):
    dqn = torch.load(network_path)
    return dqn

def create_dqn():
    num_actions = len(AllActions)
    model_layers = [256, 128, 128, 64, 32]
    obs_input_shape = len(State._fields)
    learning_rate = 1e-5
    dqn = DQNNetwork(model_layers=model_layers, observation_space_shape=obs_input_shape, num_actions=num_actions, learning_rate=learning_rate)
    """
    batch_size=512,
							 gamma=0.999,
							 eps_start=0.9,
							 eps_end=0.05,
							 eps_decay=100000,
							 tau=0.7,
							 lr=1e-5,
							 memory_len=10000):
    """
    return dqn

def create_dqn_agent(cls, map_name, map_config, main_network: DQNNetwork, checkpoint_path: Path, memory_size: int, burn_in: int, log_name: str, exploit: bool, reward_method: RewardMethod, target_network: DQNNetwork = None, **extra_agent_args):
    buffer = ExperienceReplayBuffer(memory_size=memory_size, burn_in=burn_in)
    # Update the main net every 50 agent steps, and the target network at the end of the episode
    agent_params = DQNAgentParams(epsilon=1, epsilon_decay=0.99, min_epsilon=0.01, batch_size=512, gamma=0.99, main_network_update_frequency=50, target_network_sync_frequency=-1, target_sync_mode="soft", update_tau=0.1)
    # agent_params = DQNAgentParams(epsilon=0.9, epsilon_decay=0.99, min_epsilon=0.01, batch_size=512, gamma=0.99, main_network_update_frequency=-1, target_network_sync_frequency=-1, target_sync_mode="soft", update_tau=0.5)
    agent = cls(map_name=map_name, map_config=map_config, main_network=main_network, target_network=target_network, buffer=buffer, hyperparams=agent_params, checkpoint_path=checkpoint_path, log_name=log_name, reward_method=reward_method, **extra_agent_args)

    if exploit:
        agent.exploit()
    else:
        agent.train()

    return agent

def load_or_create_dqn_agent(cls, map_name, map_config, load_agent: bool, checkpoint_path: Path, exploit: bool, memory_size: int, burn_in: int, log_name: str, load_networks_only: bool, reward_method: RewardMethod, **extra_agent_args):
    if load_networks_only:
        main_network = load_dqn(checkpoint_path / SingleDQNAgent._MAIN_NETWORK_FILE)
        target_network = load_dqn(checkpoint_path / SingleDQNAgent._TARGET_NETWORK_FILE)
        agent = create_dqn_agent(cls=cls, map_name=map_name, map_config=map_config, checkpoint_path=checkpoint_path, main_network=main_network, target_network=target_network, memory_size=memory_size, burn_in=burn_in, log_name=log_name, exploit=exploit, reward_method=reward_method, **extra_agent_args)
    elif not load_agent:
        main_network = create_dqn()
        agent = create_dqn_agent(cls=cls, map_name=map_name, map_config=map_config, checkpoint_path=checkpoint_path, main_network=main_network, memory_size=memory_size, burn_in=burn_in, log_name=log_name, exploit=exploit, reward_method=reward_method, **extra_agent_args)
    else:
        agent = load_dqn_agent(cls=cls, map_name=map_name, map_config=map_config, checkpoint_path=checkpoint_path, exploit=exploit)

    return agent

def load_random_agent(cls, map_name, map_config, checkpoint_path: Path):
    MainLogger.get().info(f"Loading agent from file {checkpoint_path}")
    agent = cls.load(checkpoint_path, map_name=map_name, map_config=map_config)

    return agent

def create_random_agent(cls, map_name, map_config, checkpoint_path: Path, memory_size: int, burn_in: int, log_name: str, reward_method: RewardMethod, **extra_agent_args):
    buffer = ExperienceReplayBuffer(memory_size=memory_size, burn_in=burn_in)
    agent = cls(map_name=map_name, map_config=map_config, log_name=log_name, buffer=buffer, checkpoint_path=checkpoint_path, reward_method=reward_method, **extra_agent_args)

    return agent


def load_or_create_random_agent(cls, map_name, map_config, load_agent: bool, memory_size: int, burn_in: int, checkpoint_path: Path, log_name: str, reward_method: RewardMethod, **extra_agent_args):
    if load_agent:
        agent = load_random_agent(cls=cls, map_name=map_name, map_config=map_config, checkpoint_path=checkpoint_path)
    else:
        agent = create_random_agent(cls=cls, map_name=map_name, map_config=map_config, memory_size=memory_size, burn_in=burn_in, checkpoint_path=checkpoint_path, log_name=log_name, reward_method=reward_method, **extra_agent_args)

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
    if len(map_config["players"]) > 1:
        for other_agent_type in map_config["players"][1:]:
            if isinstance(other_agent_type, sc2_env.Agent):
                logger.info(f"Adding random agent as opponent #{len(other_agents) + 1}#")
                other_agents.append(SingleRandomAgent(map_name=map_name, map_config=map_config, log_name=f"Random Agent {len(other_agents) + 1}", log_level=logging.ERROR, reward_method=reward_method))

    match FLAGS.agent_key:
        case "single.random":
            agent = load_or_create_random_agent(cls=SingleRandomAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, memory_size=100000, burn_in=10000, checkpoint_path=checkpoint_path, reward_method=reward_method, log_name="Main Agent - Random")
        case "single.dqn":
            agent = load_or_create_dqn_agent(SingleDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, checkpoint_path=checkpoint_path, exploit=exploit, memory_size=100000, burn_in=10000, load_networks_only=load_networks_only, reward_method=reward_method, log_name="Main Agent - SingleDQN")
        case "multi.random.base_manager":
            agent = load_or_create_random_agent(BaseManagerDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, memory_size=100000, burn_in=10000, checkpoint_path=checkpoint_path, reward_method=reward_method)
        case "multi.random.army_recruit_manager":
            agent = load_or_create_random_agent(ArmyRecruitManagerDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, memory_size=100000, burn_in=10000, checkpoint_path=checkpoint_path, reward_method=reward_method)
        case "multi.random.army_attack_manager":
            agent = load_or_create_random_agent(ArmyAttackManagerDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, memory_size=100000, burn_in=10000, checkpoint_path=checkpoint_path, reward_method=reward_method)
        case "multi.random.game_manager":
            bm_path = checkpoint_path / "base_manager"
            base_manager = load_or_create_random_agent(BaseManagerDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, memory_size=100000, burn_in=10000, checkpoint_path=bm_path, reward_method=reward_method)
            arm_path = checkpoint_path / "army_recruit_manager"
            army_recruit_manager = load_or_create_random_agent(ArmyRecruitManagerDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, memory_size=100000, burn_in=10000, checkpoint_path=arm_path, reward_method=reward_method)
            aam_path = checkpoint_path / "army_attack_manager"
            army_attack_manager = load_or_create_random_agent(ArmyAttackManagerDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, memory_size=100000, burn_in=10000, checkpoint_path=aam_path, reward_method=reward_method)
            extra_agent_args = dict(
                base_manager=base_manager,
                army_recruit_manager=army_recruit_manager,
                army_attack_manager=army_attack_manager
            )
            agent = load_or_create_random_agent(GameManagerDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, memory_size=100000, burn_in=10000, checkpoint_path=checkpoint_path, reward_method=reward_method, **extra_agent_args)
        case "multi.dqn.base_manager":
            agent = load_or_create_dqn_agent(BaseManagerDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, checkpoint_path=checkpoint_path, exploit=exploit, memory_size=10000, burn_in=1000, load_networks_only=load_networks_only, reward_method=reward_method, log_name="Main Agent - Base Manager")
        case "multi.dqn.army_recruit_manager":
            agent = load_or_create_dqn_agent(ArmyRecruitManagerDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, checkpoint_path=checkpoint_path, exploit=exploit, memory_size=10000, burn_in=1000, load_networks_only=load_networks_only, reward_method=reward_method, log_name="Main Agent - Army Manager")
        case "multi.dqn.army_attack_manager":
            agent = load_or_create_dqn_agent(ArmyAttackManagerDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, checkpoint_path=checkpoint_path, exploit=exploit, memory_size=10000, burn_in=1000, load_networks_only=load_networks_only, reward_method=reward_method, log_name="Main Agent - Attack Manager")
        case "multi.dqn.game_manager":
            bm_path = checkpoint_path / "base_manager"
            base_manager = load_or_create_dqn_agent(BaseManagerDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, checkpoint_path=bm_path, exploit=True, memory_size=10000, burn_in=1000, load_networks_only=load_networks_only, reward_method=reward_method, log_name="Main Agent - Base Manager")
            arm_path = checkpoint_path / "army_recruit_manager"
            army_recruit_manager = load_or_create_dqn_agent(ArmyRecruitManagerDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, checkpoint_path=arm_path, exploit=True, memory_size=10000, burn_in=1000, load_networks_only=load_networks_only, reward_method=reward_method, log_name="Main Agent - Army Manager")
            aam_path = checkpoint_path / "army_attack_manager"
            army_attack_manager = load_or_create_dqn_agent(ArmyAttackManagerDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, checkpoint_path=aam_path, exploit=True, memory_size=10000, burn_in=1000, load_networks_only=load_networks_only, reward_method=reward_method, log_name="Main Agent - Attack Manager")
            extra_agent_args = dict(
                base_manager=base_manager,
                army_recruit_manager=army_recruit_manager,
                army_attack_manager=army_attack_manager
            )
            agent = load_or_create_dqn_agent(GameManagerDQNAgent, map_name=map_name, map_config=map_config, load_agent=load_agent, checkpoint_path=checkpoint_path, exploit=exploit, memory_size=10000, burn_in=1000, load_networks_only=load_networks_only, reward_method=reward_method, log_name="Main Agent - Game Manager", **extra_agent_args)
        case _:
            raise RuntimeError(f"Unknown agent key {FLAGS.agent_key}")
    try:
        if FLAGS.export_stats_only:
            agent.save_stats(save_path)
            return
        finished_episodes = 0
        # We set measure_power_secs to a very high value because we want to flush emissions as we want
        tracker = OfflineEmissionsTracker(country_iso_code="ESP", experiment_id=f"global_{FLAGS.model_id}_{map_name}", measure_power_secs=3600, log_level=logging.WARNING)
        agent.set_tracker(tracker)
        max_episode_failures = 5
        current_episode_failures = 0
        tracker.start()
        if hasattr(agent, "memory_replay_ready"):
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
                                logger.info(f"Total reward for random agent {idx + 1}: {a.reward}")
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
    flags.DEFINE_integer("num_episodes", 1, "Number of episodes to play.", lower_bound=1)
    flags.DEFINE_integer("max_burnin_episodes", 200, "Meximum number of episodes to allow to use for burning replay memories in.", lower_bound=0)
    flags.DEFINE_string("model_id", default=None, help="Determines the folder inside 'models_path' to save the model to", required=False)
    flags.DEFINE_string("models_path", help="Path where checkpoints are written to/loaded from", required=False, default=DEFAULT_MODELS_PATH)
    flags.DEFINE_integer("save_frequency_episodes", default=1, help="We save the agent every X episodes.", lower_bound=1, required=False)
    flags.DEFINE_boolean("exploit", default=False, required=False, help="Use the agent in explotation mode, not for training.")
    flags.DEFINE_boolean("save_agent", default=True, required=False, help="Whether to save the agent and/or its stats.")
    flags.DEFINE_boolean("random_mode", default=False, required=False, help="Tell the agent to run in random mode. Used mostly to ensure we collect experiences.")
    flags.DEFINE_boolean("export_stats_only", default=False, required=False, help="Set it to true if you only want to load the agent and export its stats.")
    flags.DEFINE_boolean("no_visualize", default=False, required=False, help="Set it to true if you don't want to visualize the games.")
    flags.DEFINE_boolean("load_networks_only", default=False, required=False, help="Provide this flag if you want to load DQN agents, but only load its networks (no buffers, params, etc). Might be useful for curriculum training.")
    flags.DEFINE_string("log_file", default=None, required=False, help="File to save detailed logs to")
    app.run(main)
