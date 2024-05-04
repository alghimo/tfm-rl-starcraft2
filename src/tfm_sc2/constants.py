from dataclasses import dataclass

from pysc2.lib.named_array import NamedNumpyArray
from pysc2.lib.units import Protoss, Terran, Zerg


@dataclass
class Cost:
    minerals: int = 0
    vespene: int = 0
    supply: int = 0

    def can_pay(self, player: NamedNumpyArray) -> bool:
        return (
            (player.minerals >= self.minerals) and
            (player.vespene >= self.vespene) and
            (player.food_cap - player.food_used >= self.supply)
        )

class GameConstraints:
    MAX_SUPPLY_DEPOTS = 10

class SC2Costs:
    REFINERY       = Cost(minerals=50)
    SCV            = Cost(minerals=50, supply=1)
    SUPPLY_DEPOT   = Cost(minerals=100)
    COMMAND_CENTER = Cost(minerals=400)
    BARRACKS       = Cost(minerals=150)
    MARINE         = Cost(minerals=50, supply=1)

class Constants:
    STEP_REWARD = -0.01
    INVALID_ACTION_REWARD = -1.
    NO_OP_REWARD = -1.
    COMMAND_CENTER_QUEUE_LENGTH = 5
    # TODO Review places where we use this if we ever add the option to build a reactor
    BARRACKS_QUEUE_LENGTH = 5
    WORKER_UNIT_TYPES = [Terran.SCV, Protoss.Probe, Zerg.Drone, Terran.MULE]
    HARVEST_ACTIONS = [
        359, # Function.raw_ability(359, "Harvest_Gather_SCV_unit", raw_cmd_unit, 295, 3666),
        362, # Function.raw_ability(362, "Harvest_Return_SCV_quick", raw_cmd, 296, 3667),
    ]
    BUILDING_UNIT_TYPES=[
        # Terran
        # Wings of liberty
        Terran.Armory, Terran.Barracks, Terran.Bunker, Terran.CommandCenter, Terran.CommandCenterFlying,
        Terran.OrbitalCommand, Terran.OrbitalCommandFlying, Terran.PlanetaryFortress, Terran.EngineeringBay,
        Terran.Factory, Terran.FusionCore, Terran.GhostAcademy, Terran.MissileTurret, Terran.Refinery, Terran.RefineryRich,
        Terran.SensorTower, Terran.Starport, Terran.StarportFlying, Terran.SupplyDepot, Terran.SupplyDepotLowered,
        # Zerg
        # wings of liberty
        Zerg.BanelingNest, Zerg.CreepTumor, Zerg.CreepTumorBurrowed, Zerg.CreepTumorQueen, Zerg.EvolutionChamber,
        Zerg.Extractor, Zerg.ExtractorRich, Zerg.Hatchery, Zerg.Lair, Zerg.Hive, Zerg.HydraliskDen, Zerg.InfestationPit,
        Zerg.NydusNetwork, Zerg.NydusCanal, Zerg.RoachWarren, Zerg.SpawningPool, Zerg.SpineCrawler, Zerg.SpineCrawlerUprooted,
        Zerg.Spire, Zerg.GreaterSpire, Zerg.SporeCrawler, Zerg.SporeCrawlerUprooted, Zerg.UltraliskCavern,
        # Heart of the void
        Zerg.LurkerDen,
        # Protoss
        # Wings of liberty
        Protoss.Assimilator, Protoss.CyberneticsCore, Protoss.DarkShrine, Protoss.FleetBeacon, Protoss.Forge,
        Protoss.Gateway, Protoss.Nexus, Protoss.PhotonCannon, Protoss.Pylon, Protoss.RoboticsFacility,
        Protoss.RoboticsBay, Protoss.Stargate, Protoss.TemplarArchive, Protoss.TwilightCouncil, Protoss.WarpGate,
        # Legacy of the void
        Protoss.ShieldBattery,Protoss.StasisTrap
    ]
    ARMY_UNIT_TYPES=[
        # Terran
        # Wings of liberty
        Terran.Banshee, Terran.Battlecruiser, Terran.Ghost, Terran.Hellion, Terran.Marauder, Terran.Marine, Terran.Medivac,
        Terran.Raven, Terran.Reaper, Terran.SiegeTank, Terran.Thor, Terran.VikingAssault, Terran.VikingFighter, Terran.AutoTurret,
        Terran.PointDefenseDrone,
        # Heart of the swarm
        Terran.Hellbat, Terran.WidowMine, Terran.WidowMineBurrowed,
        #Legacy of the void
        Terran.Liberator, Terran.Cyclone,
        # Zerg
        # Wings of liberty
        Zerg.Corruptor, Zerg.BroodLord, Zerg.BroodLordCocoon, Zerg.Hydralisk, Zerg.HydraliskBurrowed, Zerg.Infestor, Zerg.InfestorBurrowed, Zerg.Larva,
        Zerg.Mutalisk, Zerg.Overlord, Zerg.OverlordTransport, Zerg.OverlordTransportCocoon, Zerg.Overseer, Zerg.OverseerCocoon, Zerg.OverseerOversightMode,
        Zerg.Queen, Zerg.QueenBurrowed, Zerg.Roach, Zerg.RoachBurrowed, Zerg.Ultralisk, Zerg.UltraliskBurrowed, Zerg.Zergling, Zerg.ZerglingBurrowed,
        Zerg.Baneling, Zerg.BanelingBurrowed, Zerg.BanelingCocoon, Zerg.Broodling, Zerg.BroodlingEscort, Zerg.Changeling, Zerg.ChangelingMarine,
        Zerg.ChangelingMarineShield, Zerg.ChangelingZealot, Zerg.ChangelingZergling, Zerg.ChangelingZerglingWings, Zerg.InfestedTerran, Zerg.InfestedTerranBurrowed,
        Zerg.InfestedTerranCocoon,
        # Heart of the swarm
        Zerg.SwarmHost, Zerg.SwarmHostBurrowed, Zerg.Viper,
        # Legacy of the void
        Zerg.Lurker, Zerg.LurkerBurrowed, Zerg.LurkerCocoon, Zerg.Ravager, Zerg.RavagerBurrowed, Zerg.RavagerCocoon,
        # Protoss
        # Wings of Liberty
        Protoss.Archon, Protoss.Carrier, Protoss.Colossus, Protoss.DarkTemplar, Protoss.HighTemplar, Protoss.Immortal,
        Protoss.Mothership, Protoss.Observer, Protoss.Phoenix, Protoss.Sentry, Protoss.Stalker, Protoss.VoidRay,
        Protoss.WarpPrism, Protoss.WarpPrismPhasing, Protoss.Zealot,
        # Heart of the swarm
        Protoss.Oracle, Protoss.Tempest, Protoss.MothershipCore,
        # Legacy of the void
        Protoss.Adept, Protoss.AdeptPhaseShift, Protoss.Disruptor
    ]
    COMMAND_CENTER_UNIT_TYPES=[
        Terran.CommandCenter, Terran.CommandCenterFlying,
        Zerg.Hive,
        Protoss.Nexus,
    ]
    @classmethod
    def OTHER_BUILDING_UNIT_TYPES(cls):
        if cls._OTHER_BUILDING_UNIT_TYPES is None:
            cls._OTHER_BUILDING_UNIT_TYPES = [ut for ut in cls.BUILDING_UNIT_TYPES if ut not in cls.COMMAND_CENTER_UNIT_TYPES]
        return cls._OTHER_BUILDING_UNIT_TYPES