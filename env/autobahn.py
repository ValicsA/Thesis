"""
Description

@author Ákos Valics
"""

import numpy as np

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces import Tuple
from flow.core.rewards import desired_velocity
from flow.core import rewards
from flow.envs.base import Env

ADDITIONAL_ENV_PARAMS = {
    # acceleration of autonomous vehicles
    'accel': 3.5,
    # deceleration of autonomous vehicles
    'decel': -3.5,
    # emergency braking of the vehicle
    'emer': -9,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 30
}


class Autobahn(Env):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(-float('inf'), float('inf'), shape=(15,), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        # Accelerate (Lane change to left (0), Lane change to right (1), No lane change (2)),
        # Decelerate (No lane change (3)),
        # Maintain Speed (Lane change to left (4), Lane change to right (5), No lane change (6)),
        # Emergency Brake (No lane change (7))
        speed = Discrete(4)
        # Lane change to left, Lane change to right, No lane change
        lane_change = Discrete(3)
        return Discrete(7)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        acceleration = 0
        lane_change_action = 0

        if 0 <= rl_actions < 3:
            acceleration = ADDITIONAL_ENV_PARAMS["accel"]
            if rl_actions == 0:
                lane_change_action = 1
            if rl_actions == 1:
                lane_change_action = -1
            if rl_actions == 2:
                lane_change_action = 0
        elif rl_actions == 3:
            acceleration = ADDITIONAL_ENV_PARAMS["decel"]
            lane_change_action = 0
        elif 3 < rl_actions < 7:
            acceleration = 0
            if rl_actions == 4:
                lane_change_action = 1
            if rl_actions == 5:
                lane_change_action = -1
            if rl_actions == 6:
                lane_change_action = 0
        elif rl_actions == 7:
            acceleration = ADDITIONAL_ENV_PARAMS["emer"]
            lane_change_action = 0
        else:
            print("Not valid rl_action!")

        rl_id = "rl_0"
        self.k.vehicle.apply_acceleration(rl_id, acceleration)
        self.k.vehicle.apply_lane_change(rl_id, lane_change_action)

    def get_state(self):
        """See class definition."""
        obs = {}

        # normalizing constants
        max_speed = self.k.network.max_speed()
        target_speed = desired_velocity(self)

        for rl_id in self.k.vehicle.get_rl_ids():
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_speeds = self.k.vehicle.get_lane_leaders_speed(rl_id)
            lead_headways = self.k.vehicle.get_lane_headways(rl_id)
            follower_speeds = self.k.vehicle.get_lane_followers_speed(rl_id)
            follower_headways = self.k.vehicle.get_lane_tailways(rl_id)

            if len(lead_headways) > 1:
                observation = np.array([
                    this_speed,
                    target_speed,
                    max_speed,
                    lead_headways[0], lead_headways[1], lead_headways[2],
                    lead_speeds[0], lead_speeds[1], lead_speeds[2],
                    follower_headways[0], follower_headways[1], follower_headways[2],
                    follower_speeds[0], follower_speeds[1], follower_speeds[2]
                ])
            else:
                observation = np.array([
                    this_speed,
                    target_speed,
                    max_speed,
                    lead_headways[0], lead_headways[0], lead_headways[0],
                    lead_speeds[0], lead_speeds[0], lead_speeds[0],
                    follower_headways[0], follower_headways[0], follower_headways[0],
                    follower_speeds[0], follower_speeds[0], follower_speeds[0]
                ])

            obs.update({rl_id: observation})

        return obs["rl_0"]

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            if self.env_params.evaluate:
                # reward is speed of vehicle if we are in evaluation mode
                reward = self.k.vehicle.get_speed(rl_id)
            elif kwargs['fail']:
                # reward is 0 if a collision occurred
                reward = 0
            else:
                # reward high system-level velocities
                actual_speed = self.k.vehicle.get_speed(rl_id)
                # cost1 = desired_velocity(self, fail=kwargs['fail']) / actual_speed if actual_speed > 0 else 0
                cost1 = (actual_speed / desired_velocity(self, fail=kwargs['fail'])) * 10

                # penalize small time headways
                cost2 = 0
                t_min = 1  # smallest acceptable time headway

                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(self.k.vehicle.get_headway(rl_id) / self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min) / t_min, 0)

                # weights for cost1, cost2, and cost3, respectively
                eta1, eta2 = 1.00, 0.10

                reward = max(eta1 * cost1 + eta2 * cost2, 0)

            rewards[rl_id] = reward
        return rewards

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes.
        """
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            # leader
            lead_ids = self.k.vehicle.get_lane_leaders(rl_id)
            if len(lead_ids) > 0:
                for lead_id in lead_ids:
                    self.k.vehicle.set_observed(lead_id)
            # follower
            follow_ids = self.k.vehicle.get_lane_followers(rl_id)
            if len(follow_ids) > 0:
                for follow_id in follow_ids:
                    self.k.vehicle.set_observed(follow_id)
