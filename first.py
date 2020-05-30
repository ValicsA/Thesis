"""
Description

@author Ákos Valics
"""

import os
import random
import warnings
import numpy as np

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
        inflow.add(
            veh_type=self.vehicle_types[1],
            edge="edge0",
            vehs_per_hour=self.flow_rate,
            depart_lane="free",
            depart_speed="random")
        inflow.add(
            veh_type=self.vehicle_types[2],
            edge="edge0",
            vehs_per_hour=self.flow_rate/10,
            depart_lane="first",
            depart_speed="random")

        return inflow


class HighwayNetwork(Network):

    def specify_nodes(self, net_params):
        nodes = [{"id": "node_0", "x": 300, "y": 0},
                 {"id": "node_1", "x": 500, "y": 0},
                 {"id": "node_2", "x": 1000, "y": 0},
                 {"id": "node_3", "x": 15000, "y": 0},
                 {"id": "node_4", "x": 15500, "y": 0},
                 {"id": "node_5", "x": 0, "y": -115},
                 {"id": "node_6", "x": 300, "y": -115}]
        return nodes

    def specify_edges(self, net_params):
        lanes = net_params.additional_params["num_lanes"]
        speed_limit = net_params.additional_params["speed_limit"]
        edges = [
            {
                "id": "edge0",
                "numLanes": lanes-1,
                "speed": speed_limit,
                "from": "node_0",
                "to": "node_1",
                "length": 200
            },
            {
                "id": "edge1",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "node_1",
                "to": "node_2",
                "length": 500
            },
            {
                "id": "edge2",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "node_2",
                "to": "node_3",
                "length": 14000
            },
            {
                "id": "edge3",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "node_3",
                "to": "node_4",
                "length": 500
            },
            {
                "id": "edge4",
                "numLanes": 1,
                "speed": speed_limit/10,
                "from": "node_5",
                "to": "node_6",
                "length": 300
            },
            {
                "id": "edge5",
                "numLanes": 1,
                "speed": speed_limit/10,
                "from": "node_6",
                "to": "node_1",
                "length": 230
            }
        ]
        return edges

    def specify_routes(self, net_params):
        rts = {"edge0": ["edge0", "edge1", "edge2", "edge3"],
               "edge1": ["edge1", "edge2", "edge3"],
               "edge2": ["edge2", "edge3"],
               "edge3": ["edge3"],
               "edge4": ["edge4", "edge5", "edge1", "edge2", "edge3"],
               "edge5": ["edge5", "edge1", "edge2", "edge3"]}
        return rts

    def specify_edge_starts(self):
        edge_starts = [("edge0", 300),
                       ("edge1", 500),
                       ("edge2", 1000),
                       ("edge3", 15000),
                       ("edge4", 0),
                       ("edge5", 300)]
        return edge_starts


class Vehicles:

    def __init__(self, vehicle_types, vehicle_speeds, lane_change_modes):
        self.vehicle_types = vehicle_types
        self.vehicle_speeds = vehicle_speeds
        self.lane_change_modes = lane_change_modes

    def create_vehicles(self):
        vehicles = VehicleParams()

        # Vehicle parameters https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html
        # Emission classes https://sumo.dlr.de/docs/Models/Emissions/HBEFA3-based.html
        ego_additional_params = {
            "vClass": "passenger",
            "emissionClass": "HBEFA3/PC_G_EU6",
            "guiShape": "passenger/sedan",
        }
        truck_additional_params = {
            "vClass": "trailer",
            "emissionClass": "HBEFA3/HDV_D_EU6",
            "guiShape": "truck/semitrailer",
        }

        # RL vehicles
        vehicles.add(self.vehicle_types[0],
                     acceleration_controller=(RLController, {}),
                     routing_controller=(ContinuousRouter, {}),
                     car_following_params=SumoCarFollowingParams(max_speed=self.vehicle_speeds[0],
                                                                 accel=3.5),
                     lane_change_params=SumoLaneChangeParams(lane_change_mode=self.lane_change_modes[0]),
                     num_vehicles=3,
                     additional_parameters=ego_additional_params)
        # Flow vehicles
        vehicles.add(self.vehicle_types[1],
                     acceleration_controller=(IDMController, {"v0": random.uniform(0.7, 1.1) * self.vehicle_speeds[1],
                                                              "a": 3.5}),
                     routing_controller=(ContinuousRouter, {}),
                     car_following_params=SumoCarFollowingParams(),
                     lane_change_params=SumoLaneChangeParams(lane_change_mode=self.lane_change_modes[1]),
                     num_vehicles=0)
        # Flow trucks
        vehicles.add(self.vehicle_types[2],
                     acceleration_controller=(IDMController, {"v0": random.uniform(0.7, 1) * self.vehicle_speeds[2],
                                                              "a": 1.5}),
                     routing_controller=(ContinuousRouter, {}),
                     car_following_params=SumoCarFollowingParams(),
                     lane_change_params=SumoLaneChangeParams(lane_change_mode=self.lane_change_modes[2]),
                     num_vehicles=0,
                     additional_parameters=truck_additional_params)

        return vehicles


def run_experiment(parameters):

    additional_net_params = parameters["additional_net_params"]
    flow_rate = parameters["flow_rate"]
    name = parameters["name"]
    vehicle_types = parameters["vehicle_types"]
    vehicle_speeds = parameters["vehicle_speeds"]
    lane_change_modes = parameters["lane_change_modes"]
    experiment_len = parameters["experiment_len"]
    emission_path = parameters["emission_path"]

    inflow_c = Inflow(flow_rate=flow_rate, vehicle_types=vehicle_types)
    inflow = inflow_c.create_inflow()

    net_params = NetParams(additional_params=additional_net_params, inflows=inflow)

    initial_config = InitialConfig(spacing="random", perturbation=1, edges_distribution=["edge4"])
    traffic_lights = TrafficLightParams()

    sim_params = SumoParams(sim_step=1, render=True, emission_path=emission_path, restart_instance=True,
                            overtake_right=True)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    vehicle_c = Vehicles(vehicle_types=vehicle_types, vehicle_speeds=vehicle_speeds,
                         lane_change_modes=lane_change_modes)
    vehicles = vehicle_c.create_vehicles()

    flow_params = dict(
        exp_tag=name,
        env_name=AccelEnv,
        network=HighwayNetwork,
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


def create_parameters_dict(flow_rate, traffic_speed, ego_speed, session_id):
    parameters = {
        "additional_net_params": {
            "num_lanes": 3,
            "speed_limit": 35
        },
        "flow_rate": flow_rate,
        "name": f"emission_highway_case_{session_id}",
        "vehicle_types": ["rl", "traffic", "truck"],
        "vehicle_speeds": [ego_speed, traffic_speed[0], traffic_speed[1]],
        "lane_change_modes": ["strategic", "strategic", "strategic"],
        "experiment_len": 900,
        "emission_path": "emission_results"
    }
    with open("/home/akos/workspace/Thesis/emission_results/test_cases.txt", 'a') as test_cases:
        test_cases.write(f"\ntest case {session_id}: {parameters}")
    return parameters


def main():
    """
    Runs simulations with defined conditions.
    """
    flow_rates = [1800, 1500, 1200, 1000, 800]
    # Speed of traffic vehicles and trucks
    traffic_speeds = [[25, 20], [30, 20]]
    ego_speeds = [35, 30, 25]
    i = 1
    for flow_rate in flow_rates:
        for ego_speed in ego_speeds:
            for traffic_speed in traffic_speeds:
                parameters = create_parameters_dict(flow_rate=flow_rate, traffic_speed=traffic_speed,
                                                    ego_speed=ego_speed, session_id=i)
                run_experiment(parameters)
                i += 1


if __name__ == '__main__':
    main()
    print("Move the emission results to '/home/workspace/emission_results', do not push to Github repository!")


# def main(mode):
#     """
#     Runs simulations with defined conditions.
#     :param mode: if equals "basic_simulations" 45 basic cases
#                  if equals "additional_simulations" pass
#     return: None
#     """
#     if mode == "basic_simulations":
#         # flows = [700, 500, 300, 200, 100]
#         flow = 500
#         # human_speeds = [[15, 20, 22, 25, 27, 30, 35], [10, 12, 15, 18, 20, 23, 25], [20, 23, 25, 27, 30, 33, 35]]
#         human_speeds = [15, 20, 30]
#         rl_speeds = [20, 30, 35]
#         traffic_vehicles_nums = [10, 15, 20, 25]
#         i = 1
#         for human_speed in human_speeds:
#             for rl_speed in rl_speeds:
#                 for traffic_vehicles_num in traffic_vehicles_nums:
#                     parameters = create_parameters(case_num=i, flow=flow, human_speed=human_speed, rl_speed=rl_speed, traffic_vehicles_num=traffic_vehicles_num)
#                     run_experiment(parameters)
#                     i += 1
#
#     # TODO: Update according to the new implementation of basic simulations
#     # elif mode == "additional_simulations":
#     #     # vehicles.add acceleration controller "T": 0.5, i=46
#     #     # vehicles.add acceleration controller "a": 2, i=58
#     #     # parameters "lane_change_modes": ["aggressive", "strategic", "strategic"], i=70
#     #     flows = [700, 300, 100]
#     #     human_speeds = [[20, 30], [10, 20]]
#     #     rl_speeds = [40, 30]
#     #     i = 70
#     #     for human_speed in human_speeds:
#     #         for rl_speed in rl_speeds:
#     #             for flow in flows:
#     #                 parameters = create_parameters(case_num=i, flow=flow, human_speed=human_speed, rl_speed=rl_speed)
#     #                 run_experiment(parameters)
#     #                 i += 1
#
#     else:
#         warnings.warn("Not supported simulation mode!")
#
#
# if __name__ == '__main__':
#     main(mode="basic_simulations")
#     print("Move the emission results to '/home/workspace/emission_results', do not push to Github repository!")
