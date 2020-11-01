import gym
import gym_minigrid
from gym_minigrid.wrappers import FullyObsWrapper

def make_env(env_key, seed=None):
    env = gym.make(env_key)
    # env = FullyObsWrapper(env)
    env.seed(seed)
    return env
