from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, features
from absl import app
import numpy as np
from tfm_sc2.rl.agents.agent_utils import AgentUtils

from ..utils import xy_locs, enemy_locs, self_locs

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS


class TestAgent(AgentUtils, base_agent.BaseAgent):
    def step(self, obs: TimeStep):
        super().step(obs)

        player_relative = obs.observation.feature_screen.player_relative
        roaches = xy_locs(player_relative == _PLAYER_ENEMY)
        target = roaches[np.argmax(np.array(roaches)[:, 1])]

        import pdb

        pdb.set_trace()
        if self.can_attack(obs):
            roaches = self.get_enemy_positions(obs)

            if not roaches:
                return FUNCTIONS.no_op()

            # Find the roach with max y coord.
            target = self.select_target_enemy(roaches)

            return FUNCTIONS.Attack_screen("now", target)
        else:
            print("Attack_screen.id not in available actions")

        if self.can_select_army(obs):
            # Selects all army
            print("Selecting army")
            return FUNCTIONS.select_army("select")

        import pdb

        pdb.set_trace()
        print("Observation keys: ", list(obs.observation.keys()))
        """
        Observation keys:
        [
            'single_select', 'multi_select', 'build_queue', 'cargo', 'production_queue', 'last_actions',
            'cargo_slots_available', 'home_race_requested', 'away_race_requested', 'map_name', 'feature_screen',
            'feature_minimap', 'action_result', 'alerts', 'game_loop', 'score_cumulative', 'score_by_category',
            'score_by_vital', 'player', 'control_groups', 'feature_units', 'feature_effects', 'raw_units',
            'raw_effects', 'upgrades', 'available_actions', 'radar']
        """

        print(obs.observation.feature_units._index_names)
        """
        [
            None,
            {
                'unit_type': 0, 'alliance': 1, 'health': 2, 'shield': 3, 'energy': 4, 'cargo_space_taken': 5,
                'build_progress': 6, 'health_ratio': 7, 'shield_ratio': 8, 'energy_ratio': 9, 'display_type': 10,
                'owner': 11, 'x': 12, 'y': 13, 'facing': 14, 'radius': 15, 'cloak': 16, 'is_selected': 17,
                'is_blip': 18, 'is_powered': 19, 'mineral_contents': 20, 'vespene_contents': 21,
                'cargo_space_max': 22, 'assigned_harvesters': 23, 'ideal_harvesters': 24, 'weapon_cooldown': 25,
                'order_length': 26, 'order_id_0': 27, 'order_id_1': 28, 'tag': 29, 'hallucination': 30,
                'buff_id_0': 31, 'buff_id_1': 32, 'addon_unit_type': 33, 'active': 34, 'is_on_screen': 35,
                'order_progress_0': 36, 'order_progress_1': 37, 'order_id_2': 38, 'order_id_3': 39,
                'is_in_cargo': 40, 'buff_duration_remain': 41, 'buff_duration_max': 42, 'attack_upgrade_level': 43,
                'armor_upgrade_level': 44, 'shield_upgrade_level': 45
            }
        ]
        """
        len(obs.observation.feature_units)
        # 13
        print([u.alliance for u in obs.observation.feature_units])
        print([u.is_selected for u in obs.observation.feature_units if u.alliance == 1])

        # [1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 1]
        print([u.tag for u in obs.observation.feature_units])
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        print([u.active for u in obs.observation.feature_units])
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        print(
            [
                f"x={u.x}, y={u.y}"
                for u in obs.observation.feature_units
                if u.alliance == 4
            ]
        )
        # ['x=71, y=21', 'x=71, y=28', 'x=71, y=31', 'x=71, y=24']
        print(
            [
                f"x={u.x}, y={u.y}"
                for u in obs.observation.feature_units
                if u.alliance == 4
            ]
        )
        # ['x=71, y=21', 'x=71, y=28', 'x=71, y=31', 'x=71, y=24']

        print(obs.observation.raw_units._index_names)
        """
        [
            None,
            {
                'unit_type': 0, 'alliance': 1, 'health': 2, 'shield': 3, 'energy': 4, 'cargo_space_taken': 5,
                'build_progress': 6, 'health_ratio': 7, 'shield_ratio': 8, 'energy_ratio': 9, 'display_type': 10,
                'owner': 11, 'x': 12, 'y': 13, 'facing': 14, 'radius': 15, 'cloak': 16, 'is_selected': 17,
                'is_blip': 18, 'is_powered': 19, 'mineral_contents': 20, 'vespene_contents': 21,
                'cargo_space_max': 22, 'assigned_harvesters': 23, 'ideal_harvesters': 24, 'weapon_cooldown': 25,
                'order_length': 26, 'order_id_0': 27, 'order_id_1': 28, 'tag': 29, 'hallucination': 30,
                'buff_id_0': 31, 'buff_id_1': 32, 'addon_unit_type': 33, 'active': 34, 'is_on_screen': 35,
                'order_progress_0': 36, 'order_progress_1': 37, 'order_id_2': 38, 'order_id_3': 39,
                'is_in_cargo': 40, 'buff_duration_remain': 41, 'buff_duration_max': 42, 'attack_upgrade_level': 43,
                'armor_upgrade_level': 44, 'shield_upgrade_level': 45
            }
        ]
        """
        len(obs.observation.raw_units)

        print(obs.observation.feature_units.unit_type)
        print(obs.observation.feature_units._index_names)
        self._get_units_info(obs)

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

    def _get_units_info(self, obs, player_relative: features.PlayerRelative = None):
        player_relative_map = obs.observation.feature_screen.player_relative
        if player_relative is None:
            mask = player_relative_map > 0
        else:
            mask = player_relative_map == player_relative

        unit_locations = xy_locs(mask)

        if not any(unit_locations):
            return []

        import pdb

        pdb.set_trace()
        hit_points = obs.observation.feature_screen.unit_hit_points * enemy_locs
        hit_points_ratio = (
            obs.observation.feature_screen.unit_hit_points_ratio * enemy_locs
        )
        energy = obs.observation.feature_screen.unit_energy * enemy_locs
        energy_ratio = obs.observation.feature_screen.unit_energy_ratio * enemy_locs
        shields = obs.observation.feature_screen.unit_shields * enemy_locs
        unit_shields_ratio = obs.observation.feature_screen.unit_shields * enemy_locs

    def _get_enemy_units(self, obs):
        enemy_locs = enemy_locs()

        hit_points = obs.observation.feature_screen.unit_hit_points * enemy_locs
        hit_points_ratio = (
            obs.observation.feature_screen.unit_hit_points_ratio * enemy_locs
        )
        energy = obs.observation.feature_screen.unit_energy * enemy_locs
        energy_ratio = obs.observation.feature_screen.unit_energy_ratio * enemy_locs
        shields = obs.observation.feature_screen.unit_shields * enemy_locs
        unit_shields_ratio = obs.observation.feature_screen.unit_shields * enemy_locs
