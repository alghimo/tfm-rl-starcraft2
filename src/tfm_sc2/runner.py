from absl import app
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from tfm_sc2.agents.single.test_agent import TestAgent
from tfm_sc2.sc2_config import MAP_CONFIGS, SC2_CONFIG


def main(unused_argv):
    map_name = "CollectMineralsAndGas"
    map_config = MAP_CONFIGS[map_name]
    agent = TestAgent(map_config=map_config)
    try:
        while True:
            with sc2_env.SC2Env(
                #   map_name="AbyssalReef",
                #   players=[sc2_env.Agent(sc2_env.Race.zerg),
                #            sc2_env.Bot(sc2_env.Race.random,
                #                        sc2_env.Difficulty.very_easy)],
                map_name="CollectMineralsAndGas",
                players=map_config["players"],
                **SC2_CONFIG) as env:
                # agent_interface_format=features.AgentInterfaceFormat(
                #     feature_dimensions=features.Dimensions(screen=256, minimap=64),
                #     use_raw_units=True,
                #     use_raw_actions=True
                #     ),
                # step_mul=16,
                # game_steps_per_episode=0,
                # visualize=True) as env:

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
  app.run(main)



# import argparse

# from absl import app, flags
# from pysc2.agents import base_agent
# from pysc2.env import sc2_env
# from pysc2.lib import actions, features
# from tfm_sc2.agents.single.test_agent import TestAgent
# from tfm_sc2 import sc2_config as config
# """See https://itnext.io/build-a-zerg-bot-with-pysc2-2-0-295375d2f58e
# """
# def main():
#     map_name = flags.map_name
#     agent_key = flags.agent_key

#     match agent_key:
#         case "single.test":
#             agent = TestAgent()
#         case _:
#             raise RuntimeError(f"{agent_key} creation not implemented yet")

#     if map_name not in config.MAP_CONFIGS:
#         raise RuntimeError(f"No config for map {map_name}")

#     env_config = {**config.SC2_CONFIG, **config.MAP_CONFIGS[map_name]}

#     env_args = dict(
#         map_name=map_name,
#         players=[
#             # First player - our agent
#             sc2_env.Agent(sc2_env.Race.terran),
#             # Random bot with internal AI
#             sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy),
#         ],
#         # https://itnext.io/build-a-zerg-bot-with-pysc2-2-0-295375d2f58e
#         # This gives ~150 APM - a value of 8 would give ~300APM
#         step_mul=16,
#         # 0 - Let the game run for as long as necessary
#         game_steps_per_episode=0,
#         # Optional, but useful if we want to watch the game
#         visualize=True,
#         agent_interface_format=sc2_env.parse_agent_interface_format(
#         #   feature_screen=FLAGS.feature_screen_size,
#         #   feature_minimap=FLAGS.feature_minimap_size,
#         #   rgb_screen=FLAGS.rgb_screen_size,
#         #   rgb_minimap=FLAGS.rgb_minimap_size,
#         #   action_space=FLAGS.action_space,
#           use_feature_units=True,
#           use_raw_units=True,),


#     )
#     try:
#         while True:
#             with sc2_env.SC2Env(**env_args) as env:
#                 agent.setup(env.observation_spec(), env.action_spec())
#                 timesteps = env.reset()
#                 agent.reset()
#                 while True:
#                     step_actions = [agent.step(timesteps[0])]
#                     if timesteps[0].last():
#                         break
#                     timesteps = env.step(step_actions)
#     except KeyboardInterrupt:
#         pass

# if __name__ == "__main__":
#     flags.DEFINE_enum("agent_key", "single.test", ["single.test"], "Agent to use.")
#     flags.DEFINE_enum("map_name", "Simple64", ["CollectMineralsAndGas", "Simple64",], "Map to use.")
#     app.run(main)
#     print("SC2 runner - parsing")
#     FLAGS = flags.FLAGS

#     parser = argparse.ArgumentParser(
#         prog='SC2 runner',
#         description='Run / train agents using PySC2'
#     )
#     parser.add_argument(
#         '--agent',
#         type=str,
#         choices=[
#             "single.test",
#             # "single_random",
#         ])

#     parser.add_argument(
#         '--map-name',
#         type=str,
#         choices=[
#             "CollectMineralsAndGas",
#             "Simple64",
#             # "single.random",
#         ])
#     print("parsing args")
#     args = parser.parse_args()

#     print("args type: ", type(args))
#     print("args agent: ", args.agent)
#     print("args map name: ", args.map_name)
#     print("parsing args.agent")
#     match args.agent:
#         case "single.test":
#             agent = TestAgent()
#         case _:
#             raise RuntimeError(f"{args.agent} creation not implemented yet")
#     print("running app")

#     app.run(lambda _: main(agent, args.map_name))
