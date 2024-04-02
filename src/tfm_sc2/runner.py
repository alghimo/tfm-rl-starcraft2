import argparse

from absl import app
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features

"""See https://itnext.io/build-a-zerg-bot-with-pysc2-2-0-295375d2f58e
"""
def main(agent, map_name):
    env_args = dict(
        map_name=map_name,
        players=[
            # First player - our agent
            sc2_env.Agent(sc2_env.Race.terran),
            # Random bot with internal AI
            sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy),
        ],
        # https://itnext.io/build-a-zerg-bot-with-pysc2-2-0-295375d2f58e
        # This gives ~150 APM - a value of 8 would give ~300APM
        step_mul=16,
        # 0 - Let the game run for as long as necessary
        game_steps_per_episode=0,
        # Optional, but useful if we want to watch the game
        visualize=True,
        use_feature_units=True,
        use_raw_units=True,
    )
    try:
        while True:
            with sc2_env.SC2Env(**env_args) as env:
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
    parser = argparse.ArgumentParser(
        prog='SC2 runner',
        description='Run / train agents using PySC2'
    )
    parser.add_argument(
        '-a', '--agent',
        choices=[
            "single.test",
            # "single.random",
        ])
    
    parser.add_argument(
        '-m', '--map',
        choices=[
            "CollectMineralsAndGas",
            "Simple64",
            # "single.random",
        ])
    
    app.run(main)