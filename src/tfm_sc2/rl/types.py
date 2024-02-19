from collections import namedtuple
from enum import IntEnum

from pysc2.lib.units import Neutral

Position = namedtuple("Position", ["x", "y"])
MapUnitInfo = namedtuple("MapUnitInfo", ["player_relative", "unit_type", "tag", "hit_points", "hit_points_ratio", "energy", "energy_ratio", "shields", "shields_ratio"])

class Minerals(IntEnum):
    BattleStationMineralField = Neutral.BattleStationMineralField
    BattleStationMineralField750 = Neutral.BattleStationMineralField750
    LabMineralField = Neutral.LabMineralField
    LabMineralField750 = Neutral.LabMineralField750
    MineralCrystalItem = Neutral.MineralCrystalItem
    MineralField = Neutral.Neutral.MineralField
    MineralField450 = Neutral.MineralField450
    MineralField750 = Neutral.MineralField750
    MineralPallet = Neutral.MineralPallet
    MineralShards = Neutral.MineralShards
    NaturalMineralShards = Neutral.NaturalMineralShards
    PurifierMineralField = Neutral.PurifierMineralField
    PurifierMineralField750 = Neutral.PurifierMineralField750
    PurifierRichMineralField = Neutral.PurifierRichMineralField
    PurifierRichMineralField750 = Neutral.PurifierRichMineralField750
    RichMineralField = Neutral.RichMineralField
    RichMineralField750 = Neutral.RichMineralField750

class Gas(IntEnum):
    NaturalGas = Neutral.Gas
    PickupGasPallet500 = Neutral.PickupGasPallet500
    ProtossGasCrystalItem = Neutral.ProtossGasCrystalItem
    ProtossVespeneGeyser = Neutral.ProtossVespeneGeyser
    PurifierVespeneGeyser = Neutral.PurifierVespeneGeyser
    RichVespeneGeyser = Neutral.RichVespeneGeyser
    ShakurasVespeneGeyser = Neutral.ShakurasVespeneGeyser
    SpacePlatformGeyser = Neutral.SpacePlatformGeyser
    TerranGasCanisterItem = Neutral.TerranGasCanisterItem
    VespeneGeyser = Neutral.VespeneGeyser
    ZergGasPodItem = Neutral.ZergGasPodItem

class Salvage(IntEnum):
    PickupHugeScrapSalvage = Neutral.PickupHugeScrapSalvage
    PickupMediumScrapSalvage = Neutral.PickupMediumScrapSalvage
    PickupSmallScrapSalvage = Neutral.PickupSmallScrapSalvage
    ReptileCrate = Neutral.ReptileCrate
    