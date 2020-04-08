"""Runner script for single and multi-agent reinforcement learning experiments.

This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train.py EXP_CONFIG
"""

import argparse
import json
import os
import sys
from time import strftime

from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')

    # optional input parameters
    parser.add_argument(
        '--rl_trainer', type=str, default="RLlib",
        help='the RL trainer to use. either RLlib or Stable-Baselines')

    parser.add_argument(
        '--num_cpus', type=int, default=1,
        help='How many CPUs to use')
    parser.add_argument(
        '--num_steps', type=int, default=5000,
        help='How many total steps to perform learning over')
    parser.add_argument(
        '--rollout_size', type=int, default=1000,
        help='How many steps are in a training batch.')

    return parser.parse_known_args(args)[0]


def run_model_stablebaseline(flow_params, num_cpus=1, rollout_size=50, num_steps=50):
    """Run the model for num_steps if provided.

    Parameters
    ----------
    num_cpus : int
        number of CPUs used during training
    rollout_size : int
        length of a single rollout
    num_steps : int
        total number of training steps
    The total rollout length is rollout_size.

    Returns
    -------
    stable_baselines.*
        the trained model
    """
    if num_cpus == 1:
        constructor = env_constructor(params=flow_params, version=0)()
        # The algorithms require a vectorized environment to run
        env = DummyVecEnv([lambda: constructor])
    else:
        env = SubprocVecEnv([env_constructor(params=flow_params, version=i)
                             for i in range(num_cpus)])

    train_model = DQN(MlpPolicy, env, verbose=1)
    train_model.learn(total_timesteps=num_steps)
    return train_model


def train():
    flags = parse_args(sys.argv[1:])

    # import relevant information from the exp_config script
    module = __import__("config", fromlist=[flags.exp_config])
    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
    else:
        assert False, "Unable to find experiment config!"

    if flags.rl_trainer == "Stable-Baselines":
        flow_params = submodule.flow_params
        # Path to the saved files
        exp_tag = flow_params['exp_tag']
        result_name = '{}/{}'.format(exp_tag, strftime("%Y-%m-%d-%H:%M:%S"))

        # Perform training.
        print('Beginning training.')
        model = run_model_stablebaseline(flow_params, flags.num_cpus, flags.rollout_size, flags.num_steps)

        # Save the model to a desired folder and then delete it to demonstrate
        # loading.
        print('Saving the trained model!')
        path = os.path.realpath(os.path.expanduser('~/baseline_results'))
        ensure_dir(path)
        save_path = os.path.join(path, result_name)
        model.save(save_path)

        # dump the flow params
        with open(os.path.join(path, result_name) + '.json', 'w') as outfile:
            json.dump(flow_params, outfile,
                      cls=FlowParamsEncoder, sort_keys=True, indent=4)
    else:
        assert False, "rl_trainer should be 'Stable-Baselines'!"


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


if __name__ == "__main__":
    # train()
    play_results(path="/home/akos/baseline_results/singleagent_autobahn/", result_name="2020-04-08-20:18:28")
