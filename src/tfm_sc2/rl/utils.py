from typing import List
from pysc2.lib import actions, features
from ..types import Position

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


def xy_locs(mask) -> List[Position]:
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return [Position(x, y) for x, y in zip(x, y)]


def self_locs(obs):
    return _xy_locs(obs, _PLAYER_SELF)


def enemy_locs(obs):
    return _xy_locs(obs, _PLAYER_ENEMY)


def neutral_locs(obs):
    return _xy_locs(obs, _PLAYER_NEUTRAL)
