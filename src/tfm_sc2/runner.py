from absl import app, flags
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from tfm_sc2.actions import AllActions
from tfm_sc2.agents.dqn_agent import DQNAgentParams, State
from tfm_sc2.agents.single.single_dqn_agent import SingleDQNAgent
from tfm_sc2.agents.single.single_random_agent import SingleRandomAgent
from tfm_sc2.networks.dqn_network import DQNNetwork
from tfm_sc2.networks.experience_replay_buffer import ExperienceReplayBuffer
from tfm_sc2.sc2_config import MAP_CONFIGS, SC2_CONFIG


def main(unused_argv):
    FLAGS = flags.FLAGS

    map_name = FLAGS.map_name
    if map_name not in MAP_CONFIGS:
        raise RuntimeError(f"No config for map {map_name}")
    map_config = MAP_CONFIGS[map_name]

    match FLAGS.agent_key:
        case "single.random":
            agent = SingleRandomAgent(map_name=map_name, map_config=map_config)
        case "single.dqn":
            # TODO create networks, hyperparams, etc
            num_actions = len(AllActions)
            model_layers = [64, 48, 32]
            obs_input_shape = len(State._fields)
            learning_rate = 1e-4
            dqn = DQNNetwork(model_layers=model_layers, observation_space_shape=obs_input_shape, num_actions=num_actions, learning_rate=learning_rate)

            buffer = ExperienceReplayBuffer(memory_size=50000, burn_in=5000)
            agent_params = DQNAgentParams(epsilon=0.1, epsilon_decay=0.99, min_epsilon=0.01, batch_size=32, gamma=0.99, main_network_update_frequency=1, target_network_sync_frequency=50, target_sync_mode="soft", update_tau=0.01)
            agent = SingleDQNAgent(map_name=map_name, map_config=map_config, main_network=dqn, buffer=buffer, hyperparams=agent_params)
        case _:
            raise RuntimeError(f"Unknown agent key {FLAGS.agent_key}")
    try:
        while True:
            with sc2_env.SC2Env(
                map_name=map_name,
                players=map_config["players"],
                **SC2_CONFIG) as env:

                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
                break
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    flags.DEFINE_enum("agent_key", "single.random", ["single.random", "single.dqn"], "Agent to use.")
    flags.DEFINE_enum("map_name", "Simple64", ["CollectMineralsAndGas", "Simple64", "BuildMarines"], "Map to use.")
    app.run(main)
