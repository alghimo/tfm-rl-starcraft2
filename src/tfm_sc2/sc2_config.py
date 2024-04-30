import os

from pysc2.env import sc2_env
from pysc2.lib import features, units
from tfm_sc2.actions import (
    AllActions,
    ArmyAttackManagerActions,
    ArmyRecruitManagerActions,
    BaseManagerActions,
    ResourceManagerActions,
)

SC2_CONFIG = dict(
    agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=256, minimap=64),
                    use_raw_units=True,
                    use_raw_actions=True),
    step_mul=16,
    game_steps_per_episode=0,
    visualize=True,
    disable_fog=True
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
        positions={
            "top_left": {
                units.Terran.CommandCenter: [(23, 72), (57, 31)],
                units.Terran.SupplyDepot:
                    [(17, 38), (17, 36), (17, 34), (17, 32), (17, 30)],#, (19, 29), (19, 27), (21, 27), (23, 26), (27, 26)],
                units.Terran.Barracks:
                    [(21, 41), (31, 26),] # (25, 41),
            },
            "bottom_right": {
                units.Terran.CommandCenter: [(23, 72), (57, 31)],
                units.Terran.SupplyDepot:
                    [(51, 79), (53, 79), (55, 77), (57, 76), (59, 76)], #(61, 75), (62, 73), (63, 71), (63, 69), (63, 67), (63, 65), ],
                units.Terran.Barracks:
                    [(47, 75), (56, 63),]#(52, 63),
            }
        },
        multiple_positions=True,
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Agent(sc2_env.Race.terran),
            # sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy),
        ],
        available_actions=list(AllActions)
    ),
    CollectMineralsAndGas=dict(
        map_name="CollectMineralsAndGas",
        positions={
            units.Terran.CommandCenter: [(35, 36)],
            units.Terran.SupplyDepot:
                [(x, y) for x in range(30, 35, 2) for y in range(29, 32, 2)]
                # + [(x, y) for x in range(30, 35, 2) for y in range(41, 44, 2)]
            ,
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=list(set(list(ResourceManagerActions) + list(BaseManagerActions)))
    ),
    BuildMarines=dict(
        map_name="BuildMarines",
        positions={
            # Seems like no extra CCs can be built here...
            # units.Terran.CommandCenter: [(36, 36), (32, 41)],
            units.Terran.CommandCenter: [],
            units.Terran.SupplyDepot:
                # [(x, y) for x in range(35, 38, 2) for y in range(31, 42, 2)]
                [(42, y) for y in range(29, 44, 2)]
                + [(x, 29) for x in range(27, 32, 2)]
                + [(x, 31) for x in range(27, 32, 2)]
                + [(x, 41) for x in range(27, 32, 2)]
                + [(x, 43) for x in range(27, 32, 2)]
            ,
            units.Terran.Barracks: [(36, 29), (39, 29), (36, 42), (39, 42)]
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=list(set(list(ResourceManagerActions) + list(BaseManagerActions) + list(ArmyRecruitManagerActions)))
    ),
    DefeatRoaches=dict(
        map_name="DefeatRoaches",
        positions={
            units.Terran.CommandCenter: [],
            units.Terran.SupplyDepot: [],
            units.Terran.Barracks: [],
        },
        multiple_positions=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        available_actions=list(ArmyAttackManagerActions)
    )
)
