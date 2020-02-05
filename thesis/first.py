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
# from flow.networks.highway import ADDITIONAL_NET_PARAMS

ADDITIONAL_NET_PARAMS = {
    "num_lanes": 3,
    "speed_limit": 40
}
# inflow rate at the highway
FLOW_RATE = 3000


inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="edge1",
    vehs_per_hour=FLOW_RATE,
    departLane="random",
    depart_speed="random")


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


name = "example"

vehicles = VehicleParams()


vehicles.add("human",
             acceleration_controller=(IDMController, {"v0": 15}),
             routing_controller=(ContinuousRouter, {}),
             car_following_params=SumoCarFollowingParams(),
             lane_change_params=SumoLaneChangeParams(lane_change_mode="strategic"),
             num_vehicles=0)

vehicles.add("rl",
             acceleration_controller=(IDMController, {"v0": 40}),
             routing_controller=(ContinuousRouter, {}),
             car_following_params=SumoCarFollowingParams(),
             lane_change_params=SumoLaneChangeParams(lane_change_mode="strategic"),
             num_vehicles=3)

# ADDITIONAL_NET_PARAMS["lanes"] = 2
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
net_params = NetParams(additional_params=additional_net_params, inflows=inflow)


initial_config = InitialConfig(spacing="random", perturbation=1, edges_distribution=["edge0"])

traffic_lights = TrafficLightParams()

sim_params = SumoParams(sim_step=0.1, render=True, emission_path='data', restart_instance=True, overtake_right=False)

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

flow_params = dict(
    exp_tag='ring_example',
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
flow_params['env'].horizon = 2000
exp = Experiment(flow_params)

# run the sumo simulation
_ = exp.run(1, convert_to_csv=True)

emission_location = os.path.join(exp.env.sim_params.emission_path, exp.env.network.name)
print(emission_location + '-emission.xml')

pd.read_csv(emission_location + '-emission.csv')
