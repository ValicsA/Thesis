"""
Description

@author Ákos Valics
"""

import os
import random
import warnings

import pandas as pd
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.core.experiment import Experiment
from flow.core.params import NetParams, SumoCarFollowingParams, SumoLaneChangeParams, InFlows, InitialConfig, \
    TrafficLightParams, VehicleParams, SumoParams, EnvParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS, AccelEnv
from flow.networks import Network


class Inflow:

    def __init__(self, flow_rate, vehicle_types):
        self.flow_rate = flow_rate
        self.vehicle_types = vehicle_types

    def create_inflow(self):

        inflow = InFlows()

        # for i in range(1, len(self.vehicle_types)):
        inflow.add(
            veh_type=self.vehicle_types[1],
            edge="edge0",
            vehs_per_hour=self.flow_rate,
            depart_lane="free",
            depart_speed="random")

        return inflow


class MyNetwork(Network):

    def specify_nodes(self, net_params):
        # specify the name and position (x,y) of each node
        nodes = [{"id": "node_ego0", "x": -400, "y": 0},
                 {"id": "node_ego1", "x": -200, "y": 0},
                 {"id": "node_in", "x": 0, "y": 0},
                 {"id": "node_middle_in", "x": 2000, "y": 0},
                 {"id": "node_middle_out", "x": 5000, "y": 0},
                 {"id": "node_out", "x": 5400, "y": 0}]

        return nodes

    def specify_edges(self, net_params):
        # this will let us control the number of lanes in the network
        lanes = net_params.additional_params["num_lanes"]
        # speed limit of vehicles in the network
        speed_limit = net_params.additional_params["speed_limit"]

        edges = [
            {
                "id": "edge0",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "node_ego0",
                "to": "node_ego1",
                "length": 200
            },
            {
                "id": "edge1",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "node_ego1",
                "to": "node_in",
                "length": 200
            },
            {
                "id": "edge2",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "node_in",
                "to": "node_middle_in",
                "length": 2000
            },
            {
                "id": "edge3",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "node_middle_in",
                "to": "node_middle_out",
                "length": 3000
            },
            {
                "id": "edge4",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "node_middle_out",
                "to": "node_out",
                "length": 400
            }
        ]

        return edges

    def specify_routes(self, net_params):
        rts = {"edge0": ["edge0", "edge1", "edge2", "edge3", "edge4"],
               "edge1": ["edge1", "edge2", "edge3", "edge4"],
               "edge2": ["edge2", "edge3", "edge4"],
               "edge3": ["edge3", "edge4"],
               "edge4": ["edge4"]}

        return rts

    def specify_edge_starts(self):

        edgestarts = [("edge0", -400),
                      ("edge1", -200),
                      ("edge2", 0),
                      ("edge3", 2000),
                      ("edge4", 5000)]

        return edgestarts


class Vehicles:

    def __init__(self, vehicle_types, vehicle_speeds, lane_change_modes, traffic_vehicles_num):
        self.vehicle_types = vehicle_types
        self.vehicle_speeds = vehicle_speeds
        self.lane_change_modes = lane_change_modes
        self.traffic_vehicles_num = traffic_vehicles_num

    def create_vehicles(self):
        vehicles = VehicleParams()

        # RL vehicles
        vehicles.add(self.vehicle_types[0],
                     acceleration_controller=(RLController, {}),
                     routing_controller=(ContinuousRouter, {}),
                     car_following_params=SumoCarFollowingParams(max_speed=self.vehicle_speeds[0], accel=3.5),
                     lane_change_params=SumoLaneChangeParams(lane_change_mode=self.lane_change_modes[0]),
                     num_vehicles=3)

        # Flow vehicles
        vehicles.add(self.vehicle_types[1],
                     acceleration_controller=(IDMController, {"v0": self.vehicle_speeds[1], "a": 3.5}),
                     routing_controller=(ContinuousRouter, {}),
                     car_following_params=SumoCarFollowingParams(max_speed=self.vehicle_speeds[0]),
                     lane_change_params=SumoLaneChangeParams(lane_change_mode=self.lane_change_modes[1]),
                     num_vehicles=0)

        # Traffic vehicles which are spawned at the beginning of the simulation
        for i in range(self.traffic_vehicles_num):
            vehicles.add(self.vehicle_types[1] + str(i),
                         acceleration_controller=(IDMController, {"v0": random.uniform(0.5, 1) * self.vehicle_speeds[2], "a": 3.5}),
                         routing_controller=(ContinuousRouter, {}),
                         car_following_params=SumoCarFollowingParams(),
                         lane_change_params=SumoLaneChangeParams(lane_change_mode=self.lane_change_modes[2]),
                         num_vehicles=1)

        return vehicles


def run_experiment(parameters):

    additional_net_params = parameters["additional_net_params"]
    flow_rate = parameters["flow_rate"]
    name = parameters["name"]
    vehicle_types = parameters["vehicle_types"]
    vehicle_speeds = parameters["vehicle_speeds"]
    lane_change_modes = parameters["lane_change_modes"]
    traffic_vehicles_num = parameters["traffic_vehicles_num"]
    experiment_len = parameters["experiment_len"]
    emission_path = parameters["emission_path"]

    inflow_c = Inflow(flow_rate=flow_rate, vehicle_types=vehicle_types)
    inflow = inflow_c.create_inflow()

    net_params = NetParams(additional_params=additional_net_params, inflows=inflow)
    # net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="random", perturbation=1, edges_distribution=["edge2"])
    traffic_lights = TrafficLightParams()

    sim_params = SumoParams(sim_step=1, render=True, emission_path=emission_path, restart_instance=True, overtake_right=False)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    vehicle_c = Vehicles(vehicle_types=vehicle_types, vehicle_speeds=vehicle_speeds, lane_change_modes=lane_change_modes, traffic_vehicles_num=traffic_vehicles_num)
    vehicles = vehicle_c.create_vehicles()

    flow_params = dict(
        exp_tag=name,
        env_name=AccelEnv,
        network=MyNetwork,
        simulator='traci',
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=vehicles,
        initial=initial_config,
        tls=traffic_lights
    )

    # number of time steps
    flow_params['env'].horizon = experiment_len
    exp = Experiment(flow_params)

    # run the sumo simulation
    _ = exp.run(1, convert_to_csv=True)

    emission_location = os.path.join(exp.env.sim_params.emission_path, exp.env.network.name)
    print(emission_location + '-emission.xml')

    pd.read_csv(emission_location + '-emission.csv')


def create_parameters(case_num, flow, human_speed, rl_speed, traffic_vehicles_num):
    parameters = {
        "additional_net_params": {
            "num_lanes": 3,
            "speed_limit": 40
        },
        "flow_rate": flow,
        "name": f"highway_case_{case_num}",
        "vehicle_types": ["rl", "traffic"],
        "vehicle_speeds": [rl_speed, rl_speed+8, human_speed],
        "lane_change_modes": ["strategic", "strategic", "strategic"],
        "traffic_vehicles_num": traffic_vehicles_num,
        "experiment_len": 600,
        "emission_path": "emission_results",
    }
    return parameters


def main(mode):
    """
    Runs simulations with defined conditions.
    :param mode: if equals "basic_simulations" 45 basic cases
                 if equals "additional_simulations" pass
    return: None
    """
    if mode == "basic_simulations":
        # flows = [700, 500, 300, 200, 100]
        flow = 500
        # human_speeds = [[15, 20, 22, 25, 27, 30, 35], [10, 12, 15, 18, 20, 23, 25], [20, 23, 25, 27, 30, 33, 35]]
        human_speeds = [15, 20, 30]
        rl_speeds = [20, 30, 35]
        traffic_vehicles_nums = [10, 15, 20, 25]
        i = 1
        for human_speed in human_speeds:
            for rl_speed in rl_speeds:
                for traffic_vehicles_num in traffic_vehicles_nums:
                    parameters = create_parameters(case_num=i, flow=flow, human_speed=human_speed, rl_speed=rl_speed, traffic_vehicles_num=traffic_vehicles_num)
                    run_experiment(parameters)
                    i += 1

    # TODO: Update according to the new implementation of basic simulations
    # elif mode == "additional_simulations":
    #     # vehicles.add acceleration controller "T": 0.5, i=46
    #     # vehicles.add acceleration controller "a": 2, i=58
    #     # parameters "lane_change_modes": ["aggressive", "strategic", "strategic"], i=70
    #     flows = [700, 300, 100]
    #     human_speeds = [[20, 30], [10, 20]]
    #     rl_speeds = [40, 30]
    #     i = 70
    #     for human_speed in human_speeds:
    #         for rl_speed in rl_speeds:
    #             for flow in flows:
    #                 parameters = create_parameters(case_num=i, flow=flow, human_speed=human_speed, rl_speed=rl_speed)
    #                 run_experiment(parameters)
    #                 i += 1

    else:
        warnings.warn("Not supported simulation mode!")


if __name__ == '__main__':
    main(mode="basic_simulations")
    print("Move the emission results to '/home/workspace/emission_results', do not push to Github repository!")
