"""
Description

@author Ákos Valics
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from flow.core.util import emission_to_csv
from flow.utils.registry import env_constructor
from flow.utils.rllib import get_flow_params
from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv


def play_results(path, result_name):
    print('Loading the trained model and testing it out!')
    save_path = os.path.join(path, result_name)
    model = DQN.load(save_path)
    flow_params = get_flow_params(os.path.join(path, result_name) + '.json')
    flow_params['sim'].render = True
    env_con = env_constructor(params=flow_params, version=0)()
    # The algorithms require a vectorized environment to run
    eval_env = DummyVecEnv([lambda: env_con])
    obs = eval_env.reset()
    reward = 0
    for _ in range(flow_params['env'].horizon):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = eval_env.step(action)
        reward += rewards
    print('the final reward is {}'.format(reward))


def generate_emission_csv(emission_path, emission_name):
    """
    Generates csv from emission xml. The xml path needs to be added manually!
    """
    xml_path = os.path.join(emission_path, (emission_name + ".xml"))
    emission_to_csv(xml_path)


def plot_velocity(emission_path, emission_name, save_dir="/home/akos/Pictures/DQN_emissions/"):
    values_dict = {
        "time": np.array([]),
        "CO": np.array([]),
        "CO2": np.array([]),
        "NOx": np.array([]),
        "fuel": np.array([]),
        "HC": np.array([]),
        "noise": np.array([]),
        "PMx": np.array([]),
        "speed": np.array([]),
    }
    dimensions = {
        "time": "[s]",
        "CO": "[mg/s]",
        "CO2": "[mg/s]",
        "NOx": "[mg/s]",
        "fuel": "[ml/s]",
        "HC": "[mg/s]",
        "noise": "[dB]",
        "PMx": "[mg/s]",
        "speed": "[m/s]",
    }
    emission_csv = os.path.join(emission_path, (emission_name + ".csv"))
    with open(emission_csv) as results_csv:
        results = csv.reader(results_csv, delimiter=',')
        line_count = 0
        for row in results:
            if line_count == 0:
                columns_name = row
                print(f"The measured values are: {columns_name}")
                line_count += 1
            else:
                if row[5] == "rl":
                    values_dict["time"] = np.append(values_dict["time"], float(row[0]))
                    values_dict["CO"] = np.append(values_dict["CO"], float(row[1]))
                    values_dict["CO2"] = np.append(values_dict["CO2"], float(row[3]))
                    values_dict["NOx"] = np.append(values_dict["NOx"], float(row[9]))
                    values_dict["fuel"] = np.append(values_dict["fuel"], float(row[10]))
                    values_dict["HC"] = np.append(values_dict["HC"], float(row[11]))
                    values_dict["noise"] = np.append(values_dict["noise"], float(row[15]))
                    values_dict["PMx"] = np.append(values_dict["PMx"], float(row[17]))
                    values_dict["speed"] = np.append(values_dict["speed"], float(row[18]))

    # Fix zero emissions
    for keys, values in values_dict.items():
        for i in range(1, len(values)):
            values_dict[keys][i] = values_dict[keys][i - 1] * 0.99 \
                if values_dict[keys][i] == 0 else values_dict[keys][i]

    # Plot results in time
    for r_key, r_value in values_dict.items():
        plt.plot(values_dict["time"], r_value)
        plt.xlabel("time " + dimensions["time"])
        plt.ylabel(r_key + " " + dimensions[r_key])
        plt.title("time - " + r_key)
        plt.savefig(save_dir + "time_" + r_key)
        plt.show()


def main():
    play_path = "/home/akos/baseline_results/singleagent_autobahn/"
    play_name = "DQN_28"
    emission_path = "/home/akos/workspace/Thesis/emission_results/"
    emission_name = "singleagent_autobahn_20200502-1847311588438051.2489085-emission"

    # play_results(path=play_path, result_name=play_name)
    # generate_emission_csv(emission_path=emission_path, emission_name=emission_name)
    plot_velocity(emission_path=emission_path, emission_name=emission_name)


if __name__ == '__main__':
    main()
