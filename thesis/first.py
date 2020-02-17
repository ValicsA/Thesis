"""
Description

@author Ákos Valics
"""

import os

import pandas as pd
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter
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
            edge="edge1",
            vehs_per_hour=self.flow_rate,
            depart_lane=0,
            depart_speed="random")
        inflow.add(
            veh_type=self.vehicle_types[1],
            edge="edge1",
            vehs_per_hour=self.flow_rate,
            depart_lane=1,
            depart_speed="random")
        inflow.add(
            veh_type=self.vehicle_types[1],
            edge="edge1",
            vehs_per_hour=self.flow_rate * 0.1,
            depart_lane=2,
            depart_speed="random")
        inflow.add(
            veh_type=self.vehicle_types[2],
            edge="edge1",
            vehs_per_hour=self.flow_rate * 0.1,
            depart_lane=0,
            depart_speed="random")
        inflow.add(
            veh_type=self.vehicle_types[2],
            edge="edge1",
            vehs_per_hour=self.flow_rate,
            depart_lane=1,
            depart_speed="random")
        inflow.add(
            veh_type=self.vehicle_types[2],
            edge="edge1",
            vehs_per_hour=self.flow_rate,
            depart_lane=2,
            depart_speed="random")
        return inflow


class MyNetwork(Network):

    def specify_nodes(self, net_params):
        # specify the name and position (x,y) of each node
        nodes = [{"id": "node_ego0", "x": -400, "y": 0},
                 {"id": "node_ego1", "x": -300, "y": 0},
                 {"id": "node_in", "x": 0, "y": 0},
                 {"id": "node_middle_in", "x": 200, "y": 0},
                 {"id": "node_middle_out", "x": 2200, "y": 0},
                 {"id": "node_out", "x": 2400, "y": 0}]

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
                "length": 100
            },
            {
                "id": "edge01",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "node_ego1",
                "to": "node_in",
                "length": 300
            },
            {
                "id": "edge1",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "node_in",
                "to": "node_middle_in",
                "length": 200
            },
            {
                "id": "edge2",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "node_middle_in",
                "to": "node_middle_out",
                "length": 2000
            },
            {
                "id": "edge3",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "node_middle_out",
                "to": "node_out",
                "length": 200
            }
        ]

        return edges

    def specify_routes(self, net_params):
        rts = {"edge0": ["edge0", "edge01", "edge1", "edge2", "edge3"],
               "edge01": ["edge01", "edge1", "edge2", "edge3"],
               "edge1": ["edge1", "edge2", "edge3"],
               "edge2": ["edge2", "edge3"],
               "edge3": ["edge3"]}

        return rts

    def specify_edge_starts(self):

        edgestarts = [("edge0", -400),
                      ("edge01", -100),
                      ("edge1", 0),
                      ("edge2", 200),
                      ("edge3", 2200)]

        return edgestarts


class Vehicles:

    def __init__(self, vehicle_types, vehicle_speeds, lane_change_modes):
        self.vehicle_types = vehicle_types
        self.vehicle_speeds = vehicle_speeds
        self.lane_change_modes = lane_change_modes

    def create_vehicles(self):
        vehicles = VehicleParams()

        vehicles.add(self.vehicle_types[1],
                     acceleration_controller=(IDMController, {"v0": self.vehicle_speeds[1]}),
                     routing_controller=(ContinuousRouter, {}),
                     car_following_params=SumoCarFollowingParams(),
                     lane_change_params=SumoLaneChangeParams(lane_change_mode=self.lane_change_modes[1]),
                     num_vehicles=0)

        vehicles.add(self.vehicle_types[2],
                     acceleration_controller=(IDMController, {"v0": self.vehicle_speeds[2]}),
                     routing_controller=(ContinuousRouter, {}),
                     car_following_params=SumoCarFollowingParams(),
                     lane_change_params=SumoLaneChangeParams(lane_change_mode=self.lane_change_modes[2]),
                     num_vehicles=0)

        vehicles.add(self.vehicle_types[0],
                     acceleration_controller=(IDMController, {"v0": self.vehicle_speeds[0]}),
                     routing_controller=(ContinuousRouter, {}),
                     car_following_params=SumoCarFollowingParams(),
                     lane_change_params=SumoLaneChangeParams(lane_change_mode=self.lane_change_modes[0]),
                     num_vehicles=3)
        return vehicles


def run_experiment():

    parameters = create_parameters()
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

    initial_config = InitialConfig(spacing="random", perturbation=1, edges_distribution=["edge0"])
    traffic_lights = TrafficLightParams()

    sim_params = SumoParams(sim_step=1, render=True, emission_path=emission_path, restart_instance=True, overtake_right=False)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    vehicle_c = Vehicles(vehicle_types=vehicle_types, vehicle_speeds=vehicle_speeds, lane_change_modes=lane_change_modes)
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


def create_parameters():
    parameters = {
        "additional_net_params": {
            "num_lanes": 3,
            "speed_limit": 40
        },
        "flow_rate": 100,
        "name": "highway_case_00",
        "vehicle_types": ["rl", "traffic_slow", "traffic_fast"],
        "vehicle_speeds": [40, 20, 30],
        "lane_change_modes": ["strategic", "strategic", "strategic"],
        "experiment_len": 500,
        "emission_path": "data",
    }
    return parameters


if __name__ == '__main__':
    run_experiment()
