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
import warnings

from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy, CnnPolicy, LnCnnPolicy


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser()

    # *************************
    # Required input parameters
    # *************************
    parser.add_argument(
        '--exp_config', type=str, default="singleagent_autobahn",
        help='Name of the experiment configuration file, as located in config folder!')

    # **************************************
    # Optional simulation related parameters
    # **************************************
    parser.add_argument(
        '--rl_trainer', type=str, default="Stable-Baselines",
        help='The RL trainer to use. It should be Stable-Baselines, RLlib is currently not supported!')
    parser.add_argument(
        '--num_cpus', type=int, default=1,
        help='How many CPUs to use')
    parser.add_argument(
        '--num_steps', type=int, default=500000,
        help='How many total steps to perform learning over')

    # *********************************
    # Optional model related parameters
    # *********************************
    parser.add_argument(
        '--policy', type=int, default=0,
        help='Policy. 0-MlpPolicy, 1-LnMlpPolicy, 2-CNNPolicy, 3-LnCNNPolicy')
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='Discount factor.')
    parser.add_argument(
        '--learning_rate', type=float, default=5e-4,
        help='Learning rate for adam optimizer.')
    parser.add_argument(
        '--buffer_size', type=int, default=50000,
        help='Size of the replay buffer.')
    parser.add_argument(
        '--exploration_fraction', type=float, default=0.1,
        help='Fraction of entire training period over which the exploration rate is annealed.')
    parser.add_argument(
        '--exploration_final_eps', type=float, default=0.02,
        help='Final value of random action probability.')
    parser.add_argument(
        '--exploration_initial_eps', type=float, default=1.0,
        help='Initial value of random action probability.')
    parser.add_argument(
        '--train_freq', type=int, default=1,
        help=".Update the model every train_freq steps. Set to None to disable printing.")
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Size of a batched sampled from replay buffer for training')
    parser.add_argument(
        '--double_q', type=bool, default=True,
        help='Whether to enable Double-Q learning or not.')
    parser.add_argument(
        '--learning_starts', type=int, default=100,
        help='How many steps of the model to collect transitions for before learning starts')
    parser.add_argument(
        '--target_network_update_freq', type=int, default=500,
        help='Update the target network every `target_network_update_freq` steps.')
    parser.add_argument(
        '--prioritized_replay', type=bool, default=False,
        help='If True prioritized replay buffer will be used.')
    parser.add_argument(
        '--prioritized_replay_alpha', type=float, default=0.6,
        help='Alpha parameter for prioritized replay buffer.'
             'It determines how much prioritization is used, with alpha=0 corresponding to the uniform case.')
    parser.add_argument(
        '--prioritized_replay_beta0', type=float, default=0.4,
        help='Initial value of beta for prioritized replay buffer.')
    parser.add_argument(
        '--prioritized_replay_beta_iters', type=int, default=None,
        help='Number of iterations over which beta will be annealed from initial value to 1.0.'
             'If set to None equals to max_timesteps.')
    parser.add_argument(
        '--prioritized_replay_eps', type=float, default=1e-6,
        help='Epsilon to add to the TD errors when updating priorities.')
    parser.add_argument(
        '--param_noise', type=bool, default=False,
        help='Whether or not to apply noise to the parameters of the policy.')
    parser.add_argument(
        '--dueling', type=bool, default=True,
        help='Whether to enable dueling or not!')
    parser.add_argument(
        '--verbose', type=int, default=1,
        help='The verbosity level: 0 none, 1 training information, 2 tensorflow debug.')
    parser.add_argument(
        '--tensorboard_log', type=str, default="/home/akos/baseline_results/singleagent_autobahn/logs",
        help='The log location for tensorboard (if None, no logging).')
    parser.add_argument(
        '--full_tensorboard_log', type=bool, default=True,
        help='enable additional logging when using tensorboard WARNING: this logging can take a lot of space quickly')

    return parser.parse_known_args(args)[0]


def run_model_stablebaseline(flow_params, args, model_params=None):
    """Run the model for num_steps if provided.

    Parameters
    ----------
    flow_params :
        Flow related parameters from config.
    args:
        Training arguments from parser.

    Returns
    -------
    stable_baselines.*
        the trained model
    """
    if args.num_cpus == 1:
        constructor = env_constructor(params=flow_params, version=0)()
        # The algorithms require a vectorized environment to run
        env = DummyVecEnv([lambda: constructor])
    else:
        env = SubprocVecEnv([env_constructor(params=flow_params, version=i)
                             for i in range(args.num_cpus)])
    if model_params is None:
        if args.policy == 0:
            policy = MlpPolicy
        elif args.policy == 1:
            policy = LnMlpPolicy
        elif args.policy == 2:
            policy = CnnPolicy
        elif args.policy == 3:
            policy = LnCnnPolicy
        else:
            warnings.warn("Invalid policy type! Policy set to MlpPolicy.")
            policy = MlpPolicy
        dueling = None if args.dueling else dict(dueling=False)

        train_model = DQN(policy=policy,
                          env=env,
                          gamma=args.gamma,
                          learning_rate=args.learning_rate,
                          buffer_size=args.buffer_size,
                          exploration_fraction=args.exploration_fraction,
                          exploration_final_eps=args.exploration_final_eps,
                          exploration_initial_eps=args.exploration_initial_eps,
                          train_freq=args.train_freq,
                          batch_size=args.batch_size,
                          double_q=args.double_q,
                          learning_starts=args.learning_starts,
                          target_network_update_freq=args.target_network_update_freq,
                          prioritized_replay=args.prioritized_replay,
                          prioritized_replay_alpha=args.prioritized_replay_alpha,
                          prioritized_replay_beta0=args.prioritized_replay_beta0,
                          prioritized_replay_beta_iters=args.prioritized_replay_beta_iters,
                          prioritized_replay_eps=args.prioritized_replay_eps,
                          param_noise=args.param_noise,
                          policy_kwargs=dueling,
                          verbose=args.verbose,
                          tensorboard_log=args.tensorboard_log,
                          full_tensorboard_log=args.full_tensorboard_log
                          )
    else:
        train_model = DQN(policy=model_params["policy"],
                          env=env,
                          gamma=model_params["gamma"],
                          learning_rate=model_params["learning_rate"],
                          buffer_size=model_params["buffer_size"],
                          exploration_fraction=model_params["exploration_fraction"],
                          exploration_final_eps=model_params["exploration_final_eps"],
                          exploration_initial_eps=model_params["exploration_initial_eps"],
                          train_freq=model_params["train_freq"],
                          batch_size=model_params["batch_size"],
                          double_q=model_params["double_q"],
                          learning_starts=model_params["learning_starts"],
                          target_network_update_freq=model_params["target_network_update_freq"],
                          prioritized_replay=model_params["prioritized_replay"],
                          prioritized_replay_alpha=model_params["prioritized_replay_alpha"],
                          prioritized_replay_beta0=model_params["prioritized_replay_beta0"],
                          prioritized_replay_beta_iters=model_params["prioritized_replay_beta_iters"],
                          prioritized_replay_eps=model_params["prioritized_replay_eps"],
                          param_noise=model_params["param_noise"],
                          policy_kwargs=model_params["policy_kwargs"],
                          verbose=model_params["verbose"],
                          tensorboard_log=model_params["tensorboard_log"],
                          full_tensorboard_log=model_params["full_tensorboard_log"]
                          )

    train_model.learn(total_timesteps=args.num_steps)

    return train_model


def train(model_params=None):
    args = parse_args(sys.argv[1:])

    # import relevant information from the exp_config script
    module = __import__("config", fromlist=[args.exp_config])
    if hasattr(module, args.exp_config):
        submodule = getattr(module, args.exp_config)
    else:
        assert False, "Unable to find experiment config!"

    if args.rl_trainer == "Stable-Baselines":
        flow_params = submodule.flow_params
        # Path to the saved files
        result_name = '{}/{}'.format(flow_params['exp_tag'], strftime("%Y-%m-%d-%H:%M:%S"))

        # Perform training.
        print('Beginning training.')
        model = run_model_stablebaseline(flow_params=flow_params, args=args, model_params=model_params)

        # Save the model to a desired folder and then delete it to demonstrate loading.
        print('Saving the trained model!')
        path = os.path.realpath(os.path.expanduser('~/baseline_results'))
        ensure_dir(path)
        save_path = os.path.join(path, result_name)
        model.save(save_path)

        # dump the flow params
        with open(os.path.join(path, result_name) + '.json', 'w') as outfile:
            json.dump(flow_params, outfile, cls=FlowParamsEncoder, sort_keys=True, indent=4)

    else:
        assert False, "rl_trainer should be 'Stable-Baselines'!"


def play_results(path, result_name):
    print('Loading the trained model and testing it out!')
    save_path = os.path.join(path, result_name)
    model = DQN.load(save_path)
    flow_params = get_flow_params(os.path.join(path, result_name) + '.json')
    flow_params['sim'].render = True
    flow_params['sim'].overtake_right = True
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


def main(mode):

    train() if mode == "train" else \
        play_results(path="/home/akos/baseline_results/singleagent_autobahn/", result_name="DQN_28")


if __name__ == "__main__":
    main(mode="train")
