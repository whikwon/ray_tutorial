import os
import gym

import ray
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from osim.env import ProstheticsEnv # custom environment
from ray.rllib.utils.seed import seed
from slack_notifier import slack_message


HEAD_IP_PORT = "10.0.1.8:8787"
CHECKPOINT_PATH = '/data/ray_results/baseline/checkpoint-1400'
MAX_ITERATIONS = 1000000
SAVE_PATH = '/data/ray_results/baseline'
ENV_NAME = "prosthetics"
ACCURACY = 1e-3
SEEDS = (42, 42, 42)
TOKEN = os.environ.get("SLACK_TOKEN")


def env_creator(env_config):
    env = ProstheticsEnv(False, ACCURACY)
    env.action_space = gym.spaces.Tuple([gym.spaces.Discrete(11) for _ in range(19)])
    return env

register_env(ENV_NAME, env_creator) # Register custom env

# connect to redis server.
ray.init(HEAD_IP_PORT)

agent = ppo.PPOAgent(env="prosthetics", config={
    # Discount factor
    "gamma": 0.998,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # Time horizon
    "horizon": 300,
    # Reward clipping (std)
    "clip_rewards": False,
    # Number of workers
    "num_workers": 72*4+5, # 72 * 3 + 5
    # GAE(lambda) parameter
    "lambda": 0.95,
    # Initial coefficient for KL divergence
    "kl_coeff": 0.2,
    # Number of SGD iterations in each outer loop
    "num_sgd_iter": 10,
    # Stepsize of SGD
    "sgd_stepsize": 3e-4,
    # timestep_size
    "timesteps_per_batch": 4000,
    # batch_size
    "sample_batch_size": 128,
    # Coefficient of the value function loss
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer
    "entropy_coeff": 0.001,
    # PPO clip parameter
    "clip_param": 0.2,
    # Target value for KL divergence
    "kl_target": 0.01,
    # Number of GPUs to use for SGD
    "num_gpus": 1,
    # Whether to allocate GPUs for workers (if > 0).
    "num_gpus_per_worker": 0,
    # Whether to allocate CPUs for workers (if > 0).
    "num_cpus_per_worker": 1,
    # observation preprocess
    "preprocessor_pref": "rllib",
    # Whether to rollout "complete_episodes" or "truncate_episodes"
    "batch_mode": "complete_episodes",
    # Which observation filter to apply to the observation
    "observation_filter": "MeanStdFilter",
    # Use the sync samples optimizer instead of the multi-gpu one
    "simple_optimizer": True,
    # Override model config
    "model": {
        # Whether to use LSTM model
        "use_lstm": True,
        # Max seq length for LSTM training.
        "max_seq_len": 40,
        "fcnet_hiddens": [256, 256],
        "lstm_cell_size": 256
    },
})

if CHECKPOINT_PATH is not None:
    agent.restore(CHECKPOINT_PATH)
else:
    seed(*SEEDS)

def train(iterations):
    for i in range(MAX_ITERATIONS):
        result = agent.train() # train agent a iteration
        print(result)

        if (i+1) % 50 == 0:
            agent.save(SAVE_PATH) # save agent network parameters


if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        slack_message(e, "notify", token)
