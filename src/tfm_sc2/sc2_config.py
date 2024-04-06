import os

from pysc2.env import sc2_env
from pysc2.lib import features, units
from tfm_sc2.types import Position

SC2_CONFIG = dict(
    agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=256, minimap=64),
                    use_raw_units=True,
                    use_raw_actions=True),
    step_mul=16,
    game_steps_per_episode=0,
    visualize=True
)
    # players=[sc2_env.Agent(sc2_env.Race.terran),
    #          sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy),],
    # agent_interface_format=features.AgentInterfaceFormat(
    #     action_space=actions.ActionSpace.RAW,
    #     use_raw_units=True,
    #     raw_resolution=256),
    # # This gives ~150 APM - a value of 8 would give ~300APM
    # step_mul=16,
    # # 0 - Let the game run for as long as necessary
    # game_steps_per_episode=0,
    # # Optional, but useful if we want to watch the game
    # visualize=True,
    # disable_fog=True)


MAP_CONFIGS = dict(
    Simple64=dict(
        map_name="Simple64",
    ),
    CollectMineralsAndGas=dict(
        map_name="CollectMineralsAndGas",
        positions={
            units.Terran.CommandCenter: [Position(35, 36)],
            units.Terran.SupplyDepot:
                [(x, y) for x in range(30, 36) for y in range(29, 31)]
                + [(x, y) for x in range(30, 36) for y in range(25, 27)]
            ,
        },
        players=[sc2_env.Agent(sc2_env.Race.terran)]
    )
)


