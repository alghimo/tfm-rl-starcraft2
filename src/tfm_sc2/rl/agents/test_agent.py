from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import app


_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS

def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))

def _self_locs():
    return _xy_locs(_PLAYER_SELF)

def _enemy_locs():
    return _xy_locs(_PLAYER_ENEMY)

class TestAgent(base_agent.BaseAgent):
    def step(self, obs):
        super().step(obs)

        import pdb

        pdb.set_trace()
        print("obs._fields")
        print(obs._fields)
        # ('step_type', 'reward', 'discount', 'observation')
        print("obs.step_type")
        print(obs.step_type)
        # <StepType.FIRST: 0>
        print("obs.reward")
        print(obs.reward)
        # 0.0
        print("obs.discount")
        print(obs.discount)
        # 0.0
        print("Observation keys: ", list(obs.observation.keys()))
        """
        Observation keys:
        ['single_select', 'multi_select', 'build_queue', 'cargo', 'production_queue',
         'last_actions', 'cargo_slots_available', 'home_race_requested', 'away_race_requested',
         'map_name', 'feature_screen', 'feature_minimap', 'action_result', 'alerts', 'game_loop',
         'score_cumulative', 'score_by_category', 'score_by_vital', 'player', 'control_groups',
         'upgrades', 'available_actions']
        """

        print(obs.observation.score_cumulative._index_names)
        """[{
            'score': 0, 'idle_production_time': 1, 'idle_worker_time': 2,
            'total_value_units': 3, 'total_value_structures': 4, 'killed_value_units': 5,
            'killed_value_structures': 6, 'collected_minerals': 7, 'collected_vespene': 8,
            'collection_rate_minerals': 9, 'collection_rate_vespene': 10, 'spent_minerals': 11, 'spent_vespene': 12}]
        """
        print(obs.observation.score_cumulative)
        score_cumulative = obs.observation.score_cumulative
        index_names = obs.observation.score_cumulative._index_names
        # [  0   0   0 100   0   0   0   0   0   0   0   0   0]
        score_cum_dict = {}

        for idx, k in enumerate(index_names):
            print([score_cumulative[idx] for idx, k in enumerate(index_names)])
        score_cum_dict = {k: score_cumulative[idx] for idx, k in enumerate(index_names)}

        print(obs.observation.available_actions)
        # [0 1 2 3 4 7]

        print(obs.observation.feature_minimap._index_names)
        """
        [{
            'height_map': 0, 'visibility_map': 1, 'creep': 2, 'camera': 3, 'player_id': 4,
            'player_relative': 5, 'selected': 6, 'unit_type': 7, 'alerts': 8, 'pathable': 9,
            'buildable': 10
        }, None, None]
        """
        print(obs.observation.feature_screen._index_names)
        """
        [{
            'height_map': 0, 'visibility_map': 1, 'creep': 2, 'power': 3, 'player_id': 4,
            'player_relative': 5, 'unit_type': 6, 'selected': 7, 'unit_hit_points': 8,
            'unit_hit_points_ratio': 9, 'unit_energy': 10, 'unit_energy_ratio': 11,
            'unit_shields': 12, 'unit_shields_ratio': 13, 'unit_density': 14,
            'unit_density_aa': 15, 'effects': 16, 'hallucinations': 17, 'cloaked': 18,
            'blip': 19, 'buffs': 20, 'buff_duration': 21, 'active': 22, 'build_progress': 23,
            'pathable': 24, 'buildable': 25, 'placeholder': 26
        }, None, None]
        """

        # In the DefeatRoaches minigame
        print(obs.observation.feature_screen.unit_type)
        unit_types = obs.observation.feature_screen.unit_type
        print(unit_types[19, 64])
        # 48 - Marine
        print(unit_types[33, 22])
        # 110 - Roach

        player_relative = obs.observation.feature_screen.player_relative
        from pysc2.lib.features import PlayerRelative

        print(PlayerRelative(player_relative[19, 64]))
        # 1 - PlayerRelative.SELF
        print(PlayerRelative(player_relative[33, 22]))
        # 4 - PlayerRelative.ENEMY

        print(obs.observation.feature_screen.height_map.shape)
        # (84, 84)
        for key, value in obs.observation.items():
            print(f"obs.observation['{key}']")
            print(value)
        print(obs.observation.__dict__.keys())
        # dict_keys(['single_select', 'multi_select', 'build_queue', 'cargo', 'production_queue', 'last_actions', 'cargo_slots_available', 'home_race_requested', 'away_race_requested', 'map_name', 'feature_screen', 'feature_minimap', 'action_result', 'alerts', 'game_loop', 'score_cumulative', 'score_by_category', 'score_by_vital', 'player', 'control_groups', 'upgrades', 'available_actions'])
        print(obs.observation["single_select"])
        []
        return actions.FUNCTIONS.no_op()
