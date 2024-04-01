from collections import namedtuple
from enum import IntEnum

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
