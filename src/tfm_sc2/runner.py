from absl import app, flags
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from tfm_sc2.agents.single.single_random_agent import SingleRandomAgent
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
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    flags.DEFINE_enum("agent_key", "single.random", ["single.random"], "Agent to use.")
    flags.DEFINE_enum("map_name", "Simple64", ["CollectMineralsAndGas", "Simple64", "BuildMarines"], "Map to use.")
    app.run(main)
