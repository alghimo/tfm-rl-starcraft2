from collections import namedtuple
from enum import Enum, IntEnum

from pysc2.lib.units import Neutral

Position = namedtuple("Position", ["x", "y"])
MapUnitInfo = namedtuple("MapUnitInfo", ["player_relative", "unit_type", "tag", "hit_points", "hit_points_ratio", "energy", "energy_ratio", "shields", "shields_ratio"])


class PickupItems(IntEnum):
    GasPallet500 = Neutral.PickupGasPallet500
    HugeScrapSalvage = Neutral.PickupHugeScrapSalvage
    MediumScrapSalvage = Neutral.PickupMediumScrapSalvage
    MineralCrystalItem = Neutral.MineralCrystalItem
    MineralPallet = Neutral.MineralPallet
    MineralShards = Neutral.MineralShards
    NaturalGas = Neutral.NaturalGas
    NaturalMineralShards = Neutral.NaturalMineralShards
    ProtossGasCrystalItem = Neutral.ProtossGasCrystalItem
    SmallScrapSalvage = Neutral.PickupSmallScrapSalvage
    TerranGasCanisterItem = Neutral.TerranGasCanisterItem
    ZergGasPodItem = Neutral.ZergGasPodItem

    @staticmethod
    def contains(unit):
        return unit in list(PickupItems)

class Minerals(IntEnum):
    BattleStationMineralField = Neutral.BattleStationMineralField
    BattleStationMineralField750 = Neutral.BattleStationMineralField750
    LabMineralField = Neutral.LabMineralField
    LabMineralField750 = Neutral.LabMineralField750
    MineralField = Neutral.MineralField
    MineralField450 = Neutral.MineralField450
    MineralField750 = Neutral.MineralField750
    PurifierMineralField = Neutral.PurifierMineralField
    PurifierMineralField750 = Neutral.PurifierMineralField750
    PurifierRichMineralField = Neutral.PurifierRichMineralField
    PurifierRichMineralField750 = Neutral.PurifierRichMineralField750
    RichMineralField = Neutral.RichMineralField
    RichMineralField750 = Neutral.RichMineralField750

    @staticmethod
    def contains(unit):
        return unit in list(Minerals)

class Gas(IntEnum):
    ProtossVespeneGeyser = Neutral.ProtossVespeneGeyser
    PurifierVespeneGeyser = Neutral.PurifierVespeneGeyser
    RichVespeneGeyser = Neutral.RichVespeneGeyser
    ShakurasVespeneGeyser = Neutral.ShakurasVespeneGeyser
    SpacePlatformGeyser = Neutral.SpacePlatformGeyser
    VespeneGeyser = Neutral.VespeneGeyser

    @staticmethod
    def contains(unit):
        return unit in list(Gas)

class ScvState(IntEnum):
    IDLE = 0
    BUILDING = 1
    HARVEST_MINERALS = 2
    HARVEST_GAS = 3
    MOVING = 4
    ATTACKING = 5

class RewardMethod(IntEnum):
    REWARD = 0
    SCORE = 1
    ADJUSTED_REWARD = 2

class AgentStage(Enum):
    BURN_IN = "burn-in"
    EXPLOIT = "exploit"
    TRAINING = "training"
    UNKNOWN = "UNK"

DQNAgentParams = namedtuple('DQNAgentParams',
                            field_names=["epsilon", "epsilon_decay", "min_epsilon", "batch_size", "gamma", "loss", "main_network_update_frequency", "target_network_sync_frequency", "target_sync_mode", "update_tau"],
                            defaults=(0.1, 0.99, 0.01, 32, 0.99, None, 1, 50, "soft", 0.001))

State = namedtuple('State',
                            field_names=[
                                "can_harvest_minerals", "can_recruit_worker", "can_build_supply_depot", "can_build_command_center",
                                "can_build_barracks",  "can_recruit_marine",  "can_attack",
                                # Actions available on the map
                                # "map_actions",
                                # Command centers
                                "num_command_centers", "num_completed_command_centers",
                                "command_center_0_order_length", "command_center_1_order_length", "command_center_2_order_length", "command_center_3_order_length",
                                "command_center_0_num_workers", "command_center_1_num_workers", "command_center_2_num_workers", "command_center_3_num_workers",
                                # Workers
                                "num_workers", "num_idle_workers", "pct_idle_workers", "num_mineral_harvesters", "pct_mineral_harvesters", "num_gas_harvesters",  "pct_gas_harvesters",
                                # Buildings
                                "num_refineries", "num_completed_refineries", "num_supply_depots", "num_completed_supply_depots", "num_barracks", "num_completed_barracks", "num_other_buildings",
                                # Army
                                "num_marines", "num_marines_in_queue", "num_other_army_units",
                                # Resources
                                "free_supply", "minerals", "gas",
                                # Scores
                                # Cumulative
                                "score_cumulative_score", "score_cumulative_idle_production_time", "score_cumulative_idle_worker_time",
                                "score_cumulative_total_value_units", "score_cumulative_total_value_structures", "score_cumulative_killed_value_units",
                                "score_cumulative_killed_value_structures", "score_cumulative_collected_minerals", "score_cumulative_collected_vespene",
                                "score_cumulative_collection_rate_minerals", "score_cumulative_collection_rate_vespene", "score_cumulative_spent_minerals",
                                "score_cumulative_spent_vespene",
                                # By category
                                "score_food_used_none", "score_food_used_army", "score_food_used_economy", "score_food_used_technology", "score_food_used_upgrade",
                                "score_killed_minerals_none", "score_killed_minerals_army", "score_killed_minerals_economy", "score_killed_minerals_technology", "score_killed_minerals_upgrade",
                                "score_killed_vespene_none", "score_killed_vespene_army", "score_killed_vespene_economy", "score_killed_vespene_technology", "score_killed_vespene_upgrade",
                                "score_lost_minerals_none", "score_lost_minerals_army", "score_lost_minerals_economy", "score_lost_minerals_technology", "score_lost_minerals_upgrade",
                                "score_lost_vespene_none", "score_lost_vespene_army", "score_lost_vespene_economy", "score_lost_vespene_technology", "score_lost_vespene_upgrade",
                                "score_friendly_fire_minerals_none", "score_friendly_fire_minerals_army", "score_friendly_fire_minerals_economy", "score_friendly_fire_minerals_technology", "score_friendly_fire_minerals_upgrade",
                                "score_friendly_fire_vespene_none", "score_friendly_fire_vespene_army", "score_friendly_fire_vespene_economy", "score_friendly_fire_vespene_technology", "score_friendly_fire_vespene_upgrade",
                                "score_used_minerals_none", "score_used_minerals_army", "score_used_minerals_economy", "score_used_minerals_technology", "score_used_minerals_upgrade",
                                "score_used_vespene_none","score_used_vespene_army", "score_used_vespene_economy", "score_used_vespene_technology", "score_used_vespene_upgrade",
                                "score_total_used_minerals_none", "score_total_used_minerals_army", "score_total_used_minerals_economy", "score_total_used_minerals_technology", "score_total_used_minerals_upgrade",
                                "score_total_used_vespene_none", "score_total_used_vespene_army", "score_total_used_vespene_economy", "score_total_used_vespene_technology", "score_total_used_vespene_upgrade",
                                # Score by vital
                                "score_by_vital_total_damage_dealt_life", "score_by_vital_total_damage_dealt_shields", "score_by_vital_total_damage_dealt_energy",
                                "score_by_vital_total_damage_taken_life", "score_by_vital_total_damage_taken_shields", "score_by_vital_total_damage_taken_energy",
                                "score_by_vital_total_healed_life", "score_by_vital_total_healed_shields", "score_by_vital_total_healed_energy",
                                # Neutral units
                                "num_minerals", "num_geysers",
                                # Enemy info
                                "enemy_num_buildings", "enemy_total_building_health", "enemy_num_workers", "enemy_num_army_units", "enemy_total_army_health",
                            ])
