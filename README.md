### 1. Train my environment using ray  
Four steps are needed to train my own environment
1. Register my environment 
```python
from ray.tune.registry import register_env

ENV_NAME = "My_own_env"

def env_creator(env_config):
    env = ProstheticsEnv(False, ACCURACY)
    env.action_space = gym.spaces.Tuple([gym.spaces.Discrete(11) for _ in range(19)])
    return env

register_env(ENV_NAME, env_creator)
```

2. Import the agents I want
- If you want to change the config options, please check ![agents.py](https://github.com/ray-project/ray/tree/master/python/ray/rllib/agents/agents.py), ![ppo.py](https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/ppo/ppo.py)
```python
from ray.rllib.agents import ppo

agent = ppo.PPOAgent(env="My_own_env", config={...})
```

3. If you want to restore the checkpoint

```python
CHECKPOINT_PATH = "./checkpoint-100"
agent.restore(CHECKPOINT_PATH)
```

4. Train and save!
```python
NUM_ITERATIONS = 100
SAVE_PATH = "~/ray_results"
for i in range(NUM_ITERATIONS):
    agent.train()
    
    if (i+1) % 50 == 0:
        agent.save(SAVE_PATH)
```

### 2. Setting cluster for distributed training
- Assume anaconda environment has all set in the VMs(head, workers).
- When use 2 VMs(head, worker)
```
$HEAD ray start --head --redis-port=8787
$WORKER ray start --redis-address=<head-ip>:8787
python ppo.py
```

- When use more than 2 VMs(head, n workers), It would be boring to execute all the VMs seperately. Let's use pssh.
```
cd remote
sh start_train.sh
```

### 3. Log checking
- `tail -f /tmp/raylogs/*`: warning, error logs during training.
- `tensorboard --logdir=~/ray_results --port=6060`: training results.

### 4. Set slack alarmer to notify the error.
- Slack notification tutorial: https://medium.com/@harvitronix/using-python-slack-for-quick-and-easy-mobile-push-notifications-5e5ff2b80aad
- `export SLACK_TOKEN=<token-you-got-from-slack>`
- Add code below to let me notified when error occurred.
```python
try:
    train()
except Exception as e:
    slack_message(e, "notify", token)
```
