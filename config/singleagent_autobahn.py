"""
Description

@author Ákos Valics
"""
import random

from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.core.params import NetParams, SumoCarFollowingParams, SumoLaneChangeParams, InFlows, InitialConfig, \
    TrafficLightParams, VehicleParams, SumoParams, EnvParams
from flow.networks import Network

from Thesis.env.autobahn import Autobahn


class Inflow:

    def __init__(self, flow_rate, vehicle_types):
        self.flow_rate = flow_rate
        self.vehicle_types = vehicle_types

    def create_inflow(self):

        inflows = InFlows()
        inflows.add(
            veh_type=self.vehicle_types[1],
            edge="edge0",
            vehs_per_hour=self.flow_rate,
            depart_lane="free",
            depart_speed="random")
        inflows.add(
            veh_type=self.vehicle_types[2],
            edge="edge0",
            vehs_per_hour=self.flow_rate/10,
            depart_lane="first",
            depart_speed="random")
        return inflows


class AutobahnNetwork(Network):

    def specify_nodes(self, net_params):
        nodes = [{"id": "node_0", "x": 300, "y": 0},
                 {"id": "node_1", "x": 500, "y": 0},
                 {"id": "node_2", "x": 1000, "y": 0},
                 {"id": "node_3", "x": 11000, "y": 0},
                 {"id": "node_4", "x": 11500, "y": 0},
                 {"id": "node_5", "x": 0, "y": -115},
                 {"id": "node_6", "x": 300, "y": -115}]
        return nodes

    def specify_edges(self, net_params):
        lanes = net_params.additional_params["num_lanes"]
        speed_limit = net_params.additional_params["speed_limit"]
        edges = [
            {
                "id": "edge0",
                "numLanes": lanes - 1,
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
                "length": 10000
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
                "speed": speed_limit / 6,
                "from": "node_5",
                "to": "node_6",
                "length": 300
            },
            {
                "id": "edge5",
                "numLanes": 1,
                "speed": speed_limit / 6,
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
                       ("edge3", 21000),
                       ("edge4", 0),
                       ("edge5", 300)]
        return edge_starts

    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        pass


class Vehicles:

    def __init__(self, vehicle_types, vehicle_speeds, lane_change_modes, additional_vehicle_params):
        self.vehicle_types = vehicle_types
        self.vehicle_speeds = vehicle_speeds
        self.lane_change_modes = lane_change_modes
        self.additional_vehicle_params = additional_vehicle_params

    def create_vehicles(self):
        vehicles = VehicleParams()

        # RL vehicle
        vehicles.add(self.vehicle_types[0],
                     acceleration_controller=(RLController, {}),
                     routing_controller=(ContinuousRouter, {}),
                     car_following_params=SumoCarFollowingParams(max_speed=self.vehicle_speeds[0], accel=3.5),
                     lane_change_params=SumoLaneChangeParams(lane_change_mode=self.lane_change_modes[0]),
                     num_vehicles=1,
                     additional_parameters=self.additional_vehicle_params["rl_additional_params"])

        # Flow vehicles
        vehicles.add(self.vehicle_types[1],
                     acceleration_controller=(IDMController, {"v0": random.uniform(0.7, 1) * self.vehicle_speeds[1],
                                                              "a": 3.5}),
                     routing_controller=(ContinuousRouter, {}),
                     car_following_params=SumoCarFollowingParams(),
                     lane_change_params=SumoLaneChangeParams(lane_change_mode=self.lane_change_modes[1]),
                     num_vehicles=0,
                     additional_parameters=self.additional_vehicle_params["flow_additional_params"])
        # Flow trucks
        vehicles.add(self.vehicle_types[2],
                     acceleration_controller=(IDMController, {"v0": random.uniform(0.7, 1) * self.vehicle_speeds[2],
                                                              "a": 1.5}),
                     routing_controller=(ContinuousRouter, {}),
                     car_following_params=SumoCarFollowingParams(),
                     lane_change_params=SumoLaneChangeParams(lane_change_mode=self.lane_change_modes[2]),
                     num_vehicles=0,
                     additional_parameters=self.additional_vehicle_params["truck_additional_params"])

        return vehicles


def create_parameters():

    parameters = {
        "additional_net_params": {"num_lanes": 3,
                                  "speed_limit": 40},

        "additional_env_params": {
            # acceleration of autonomous vehicles
            'accel': 3.5,
            # deceleration of autonomous vehicles
            'decel': -3.5,
            # emergency braking of the vehicle
            'emer': -9,
            # desired velocity for all vehicles in the network, in m/s
            "target_velocity": 40},

        "additional_vehicle_params": {
            "rl_additional_params": {
                "vClass": "evehicle",
                "emissionClass": "HBEFA3/PC_D_EU6",
                "guiShape": "passenger/sedan"},

            "flow_additional_params": {
                "vClass": "passenger",
                "emissionClass": "HBEFA3/PC_D_EU6",
                "guiShape": "passenger/sedan"},

            "truck_additional_params": {
                "vClass": "trailer",
                "emissionClass": "HBEFA3/HDV_D_EU6",
                "guiShape": "truck/semitrailer"},
        },
        "flow_rate": 1800,
        "name": "singleagent_autobahn",
        "vehicle_types": ["rl", "traffic", "truck"],
        "vehicle_speeds": [35, 25, 20],
        "lane_change_modes": ["strategic", "strategic", "strategic"],
        "experiment_len": 600 + 200,
        "emission_path": "emission_results",
        "ego_initial_spacing": "random",
        "ego_initial_edge": ["edge4"],
        "simulation_step_size": 1,
    }
    return parameters


params = create_parameters()

inflow_c = Inflow(flow_rate=params["flow_rate"], vehicle_types=params["vehicle_types"])
inflow = inflow_c.create_inflow()

network_params = NetParams(additional_params=params["additional_net_params"], inflows=inflow)

initial_config = InitialConfig(spacing=params["ego_initial_spacing"], edges_distribution=params["ego_initial_edge"])
traffic_lights = TrafficLightParams()

simulation_params = SumoParams(sim_step=params["simulation_step_size"], render=False,
                               emission_path=params["emission_path"], restart_instance=True, overtake_right=True)

environment_params = EnvParams(additional_params=params["additional_env_params"], horizon=params["experiment_len"])

vehicles = Vehicles(vehicle_types=params["vehicle_types"], vehicle_speeds=params["vehicle_speeds"],
                    lane_change_modes=params["lane_change_modes"],
                    additional_vehicle_params=params["additional_vehicle_params"])
vehicle_params = vehicles.create_vehicles()

flow_params = dict(
    exp_tag=params["name"],
    env_name=Autobahn,
    network=AutobahnNetwork,
    simulator='traci',
    sim=simulation_params,
    env=environment_params,
    net=network_params,
    veh=vehicle_params,
    initial=initial_config
)
