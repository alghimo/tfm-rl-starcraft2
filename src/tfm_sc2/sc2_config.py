from absl import app, logging
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
from pysc2.env import sc2_env, run_loop, environment
from main_agent import RandomAgent, DRLReshapeAgent, SparseAgent, Hyperparams
from ddqn_move import DDQNMoveAgent
import os


SC2_CONFIG = dict(
    players=[sc2_env.Agent(sc2_env.Race.terran), 
             sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy),],
    agent_interface_format=features.AgentInterfaceFormat(
        action_space=actions.ActionSpace.RAW,
        use_raw_units=True,
        raw_resolution=256),
    # This gives ~150 APM - a value of 8 would give ~300APM
    step_mul=16,
    # 0 - Let the game run for as long as necessary
    game_steps_per_episode=0,
    # Optional, but useful if we want to watch the game
    visualize=True,
    disable_fog=True)


MAP_CONFIGS = dict(
    "Simple64"=dict(
        map_name="Simple64",
    ),
    "CollectMineralsAndGas"=dict(
        map_name="CollectMineralsAndGas",
    )
)


    