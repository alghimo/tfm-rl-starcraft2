import logging
from pathlib import Path

from absl import app, flags
from codecarbon import OfflineEmissionsTracker
from pysc2.env import sc2_env
from pysc2.lib.remote_controller import ConnectError
from tfm_sc2.actions import (
    AllActions,
    ArmyAttackManagerActions,
    ArmyRecruitManagerActions,
    BaseManagerActions,
    ResourceManagerActions,
)
from tfm_sc2.agents.dqn_agent import DQNAgentParams, State
from tfm_sc2.agents.single.single_dqn_agent import SingleDQNAgent
from tfm_sc2.agents.single.single_random_agent import SingleRandomAgent
from tfm_sc2.networks.dqn_network import DQNNetwork
from tfm_sc2.networks.experience_replay_buffer import ExperienceReplayBuffer
from tfm_sc2.sc2_config import MAP_CONFIGS, SC2_CONFIG


def main(unused_argv):
    FLAGS = flags.FLAGS
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
    if load_agent:
        print(f"Agent will be loaded from file: {agent_file}")
    else:
        print(f"A new agent will be created")
    save_agent = True
    exploit = FLAGS.exploit
    # We will still save the stats when exploiting, but in a subfolder
    if exploit:
        save_path = checkpoint_path / "exploit"
        save_path.mkdir(exist_ok=True, parents=True)

    print(f"Map name: {map_name}")
    print(f"Map available actions: ", map_config["available_actions"])

    other_agents = []
    if len(map_config["players"]) > 1:
        for other_agent_type in map_config["players"][1:]:
            if isinstance(other_agent_type, sc2_env.Agent):
                print(f"Adding random agent as opponent #{len(other_agents) + 1}#")
                other_agents.append(SingleRandomAgent(map_name=map_name, map_config=map_config, log_name=f"Random Agent {len(other_agents) + 1}", log_level=logging.ERROR))

    match FLAGS.agent_key:
        case "single.random":
            if load_agent:
                # TODO implement load method into base agent
                agent = SingleRandomAgent.load(checkpoint_path=checkpoint_path, map_name=map_name, map_config=map_config)
            else:
                agent = SingleRandomAgent(map_name=map_name, map_config=map_config, log_name = "Main Agent")
        case "single.dqn":
            if load_agent:
                print(f"Loading agent from file {checkpoint_path}")
                agent = SingleDQNAgent.load(checkpoint_path, map_name=map_name, map_config=map_config)
            else:
                num_actions = len(AllActions)
                model_layers = [64, 48, 32]
                obs_input_shape = len(State._fields)
                learning_rate = 1e-4
                dqn = DQNNetwork(model_layers=model_layers, observation_space_shape=obs_input_shape, num_actions=num_actions, learning_rate=learning_rate)

                buffer = ExperienceReplayBuffer(memory_size=100000, burn_in=10000)
                agent_params = DQNAgentParams(epsilon=0.1, epsilon_decay=0.99, min_epsilon=0.01, batch_size=32, gamma=0.99, main_network_update_frequency=1, target_network_sync_frequency=50, target_sync_mode="soft", update_tau=0.01)
                agent = SingleDQNAgent(map_name=map_name, map_config=map_config, main_network=dqn, buffer=buffer, hyperparams=agent_params, log_name = "Main Agent")

            if exploit:
                agent.exploit()
            else:
                agent.train()
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
                                print(f"Total reward for random agent {idx + 1}: {a.reward}")
                    current_episode_failures = 0
            except ConnectError as error:
                print("Couldn't connect to SC2 environment, trying to restart the episode again")
                print(error)
                finished_episodes -= 1
                current_episode_failures += 1
                if current_episode_failures >= max_episode_failures:
                    print(f"Reached max number of allowed episode failures, stopping run")
                    break
            finally:
                finished_episodes += 1
                if save_agent and (finished_episodes % FLAGS.save_frequency_episodes) == 0:
                    print(f"Saving agent after {finished_episodes} episodes")
                    agent.save(save_path)
                    already_saved = True
        if save_agent and not already_saved:
            print(f"Saving final agent after {finished_episodes} episodes")
            total_emissions = tracker.stop()
            agent.save(save_path)
        else:
            total_emissions = tracker.stop()
            print(f"Total emissions after {finished_episodes} episodes for agent {agent._log_name} (and {len(other_agents)} other agents): {total_emissions:.2f}")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    DEFAULT_MODELS_PATH = str(Path(__file__).parent.parent.parent / "models")

    flags.DEFINE_enum("agent_key", "single.random", ["single.random", "single.dqn"], "Agent to use.")
    flags.DEFINE_enum("map_name", "Simple64", ["CollectMineralsAndGas", "Simple64", "BuildMarines", "DefeatRoaches", "DefeatZerglingsAndBanelings"], "Map to use.")
    flags.DEFINE_integer("num_episodes", 1, "Number of episodes to play.", lower_bound=1)
    flags.DEFINE_string("model_id", default=None, help="Determines the folder inside 'models_path' to save the model to", required=False)
    flags.DEFINE_string("models_path", help="Path where checkpoints are written to/loaded from", required=False, default=DEFAULT_MODELS_PATH)
    flags.DEFINE_integer("save_frequency_episodes", default=1, help="We save the agent every X episodes.", lower_bound=1, required=False)
    flags.DEFINE_boolean("exploit", default=False, required=False, help="Use the agent in explotation mode, not for training.")
    flags.DEFINE_boolean("save_agent", default=True, required=False, help="Whether to save the agent and/or its stats.")
    flags.DEFINE_boolean("random_mode", default=False, required=False, help="Tell the agent to run in random mode. Used mostly to ensure we collect experiences.")
    flags.DEFINE_boolean("export_stats_only", default=False, required=False, help="Set it to true if you only want to load the agent and export its stats.")
    flags.DEFINE_boolean("no_visualize", default=False, required=False, help="Set it to true if you don't want to visualize the games.")

    app.run(main)
