from enum import IntEnum


class AllActions(IntEnum):
    NO_OP = 0
    # Resource manager actions
    HARVEST_MINERALS = 11
    COLLECT_GAS      = 12
    BUILD_REFINERY   = 13
    # Base Manager actions
    RECRUIT_SCV          = 21
    BUILD_SUPPLY_DEPOT   = 22
    BUILD_COMMAND_CENTER = 24
    # ArmyRecruit Manager actions
    RECRUIT_MARINE = 31
    RECRUIT_DOCTOR = 32
    BUILD_TECH_LAB = 33
    BUILD_BARRACKS = 34
    # ArmyAttack Manager actions

    ATTACK_UNIT_HALF_ARMY     = 41
    ATTACK_BUILDING_HALF_ARMY = 42
    ATTACK_UNIT_FULL_ARMY     = 43
    ATTACK_BUILDING_FULL_ARMY = 44

class GameManagerActions(IntEnum):
    GATHER_RESOURCES = 10
    EXPAND_BASE = 20
    EXPAND_ARMY = 30
    ATTACK = 40

class ResourceManagerActions(IntEnum):
    NO_OP            = AllActions.NO_OP
    HARVEST_MINERALS = AllActions.HARVEST_MINERALS
    COLLECT_GAS      = AllActions.COLLECT_GAS
    BUILD_REFINERY   = AllActions.BUILD_REFINERY

class BaseManagerActions(IntEnum):
    NO_OP                = AllActions.NO_OP
    RECRUIT_SCV          = AllActions.RECRUIT_SCV
    BUILD_SUPPLY         = AllActions.BUILD_SUPPLY_DEPOT
    BUILD_COMMAND_CENTER = AllActions.BUILD_COMMAND_CENTER

class ArmyRecruitManagerActions(IntEnum):
    NO_OP          = AllActions.NO_OP
    RECRUIT_MARINE = AllActions.RECRUIT_MARINE
    RECRUIT_DOCTOR = AllActions.RECRUIT_DOCTOR
    BUILD_TECH_LAB = AllActions.BUILD_TECH_LAB
    BUILD_BARRACKS = AllActions.BUILD_BARRACKS

class ArmyAttackManagerActions(IntEnum):
    NO_OP                     = AllActions.NO_OP
    ATTACK_UNIT_HALF_ARMY     = AllActions.ATTACK_UNIT_HALF_ARMY
    ATTACK_BUILDING_HALF_ARMY = AllActions.ATTACK_BUILDING_HALF_ARMY
    ATTACK_UNIT_FULL_ARMY     = AllActions.ATTACK_UNIT_FULL_ARMY
    ATTACK_BUILDING_FULL_ARMY = AllActions.ATTACK_BUILDING_FULL_ARMY
