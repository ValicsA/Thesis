"""
Description

@author Ákos Valics
"""
import Thesis.train
from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy, CnnPolicy, LnCnnPolicy


def choose_dict(mode):
    test_cases = {
        "policy": [MlpPolicy, LnMlpPolicy, CnnPolicy, LnCnnPolicy, MlpPolicy, CnnPolicy, MlpPolicy, CnnPolicy],
        "gamma": [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
        "learning_rate": [5e-4, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4],
        "buffer_size": [50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000],
        "exploration_fraction": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "exploration_final_eps": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        "exploration_initial_eps": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "train_freq": 1,
        "batch_size": [32, 32, 32, 32, 32, 32, 32, 32],
        "double_q": [True, True, True, True, False, False, True, True],
        "learning_starts": [100, 100, 100, 100, 100, 100, 100, 100],
        "target_network_update_freq": [500, 500, 500, 500, 500, 500, 500, 500],
        "prioritized_replay": [False, False, False, False, False, False, True, True],
        "prioritized_replay_alpha": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        "prioritized_replay_beta0": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        "prioritized_replay_beta_iters": [None, None, None, None, None, None, None, None],
        "prioritized_replay_eps": [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
        "param_noise": [False, False, False, False, False, False, False, False],
        "policy_kwargs": [dict(dueling=False), dict(dueling=False), dict(dueling=False), dict(dueling=False),
                          None, None, None, None],
        "verbose": 1,
        "tensorboard_log": "/home/akos/baseline_results/singleagent_autobahn/logs",
        "full_tensorboard_log": True
    }
    hyperparam_opt = {
        "policy": [],
        "gamma": [],
        "learning_rate": [],
        "buffer_size": [],
        "exploration_fraction": [],
        "exploration_final_eps": [],
        "exploration_initial_eps": [],
        "train_freq": 1,
        "batch_size": [],
        "double_q": [],
        "learning_starts": [],
        "target_network_update_freq": [],
        "prioritized_replay": [],
        "prioritized_replay_alpha": [],
        "prioritized_replay_beta0": [],
        "prioritized_replay_beta_iters": [],
        "prioritized_replay_eps": [],
        "param_noise": [],
        "policy_kwargs": [],
        "verbose": 1,
        "tensorboard_log": "/home/akos/baseline_results/singleagent_autobahn/logs",
        "full_tensorboard_log": True
    }
    if mode == "hyperparam":
        return hyperparam_opt
    elif mode == "run_tests":
        return test_cases
    else:
        print("Invalid mode."
              "Please select 'hyperparam' for hyperparameter optimization or 'run_tests' to run test cases!")
        return None


def main(mode):
    test_dict = choose_dict(mode=mode)
    num_test_cases = len(test_dict["policy"])
    for i in range(num_test_cases):
        model_params = {
            "policy": test_dict["policy"][i],
            "gamma": test_dict["gamma"][i],
            "learning_rate": test_dict["learning_rate"][i],
            "buffer_size": test_dict["buffer_size"][i],
            "exploration_fraction": test_dict["exploration_fraction"][i],
            "exploration_final_eps": test_dict["exploration_final_eps"][i],
            "exploration_initial_eps": test_dict["exploration_initial_eps"][i],
            "train_freq": test_dict["train_freq"],
            "batch_size": test_dict["batch_size"][i],
            "double_q": test_dict["double_q"][i],
            "learning_starts": test_dict["learning_starts"][i],
            "target_network_update_freq": test_dict["target_network_update_freq"][i],
            "prioritized_replay": test_dict["prioritized_replay"][i],
            "prioritized_replay_alpha": test_dict["prioritized_replay_alpha"][i],
            "prioritized_replay_beta0": test_dict["prioritized_replay_beta0"][i],
            "prioritized_replay_beta_iters": test_dict["prioritized_replay_beta_iters"][i],
            "prioritized_replay_eps": test_dict["prioritized_replay_eps"][i],
            "param_noise": test_dict["param_noise"][i],
            "policy_kwargs": test_dict["policy_kwargs"][i],
            "verbose": test_dict["verbose"],
            "tensorboard_log": test_dict["tensorboard_log"],
            "full_tensorboard_log": test_dict["full_tensorboard_log"]
        }
        Thesis.train.train(model_params=model_params)


if __name__ == '__main__':
    main(mode="run_tests")
