from enum import IntEnum


class GenericActions(IntEnum):
    NO_OP = 0

class GameManagerActions(IntEnum):
    GATHER_RESOURCES = 10
    EXPAND_BASE = 20
    EXPAND_ARMY = 30
    ATTACK = 40

class ResourceManagerActions(IntEnum):
    NO_OP            = GenericActions.NO_OP
    HARVEST_MINERALS = 11
    COLLECT_GAS      = 12
    BUILD_REFINERY   = 13

class BaseManagerActions(IntEnum):
    NO_OP                = GenericActions.NO_OP
    RECRUIT_SCV          = 21
    BUILD_SUPPLY         = 22
    BUILD_COMMAND_CENTER = 24

class ArmyRecruitManagerActions(IntEnum):
    NO_OP          = GenericActions.NO_OP
    RECRUIT_MARINE = 31
    RECRUIT_DOCTOR = 32
    BUILD_TECH_LAB = 33
    BUILD_BARRACKS = 34

class ArmyAttackManagerActions(IntEnum):
    NO_OP          = GenericActions.NO_OP
    ATTACK_UNIT_HALF_ARMY     = 31
    ATTACK_BUILDING_HALF_ARMY = 32
    ATTACK_UNIT_FULL_ARMY     = 33
    ATTACK_BUILDING_FULL_ARMY = 34
