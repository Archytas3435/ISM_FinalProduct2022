import numpy as np
import gym
from collections import deque
import math

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps
    def reset(self):
        obs = self.env.reset()
        obs = np.expand_dims(obs, 0)
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.expand_dims(obs, 0)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info
    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def _epsilon_schedule(eps_start, eps_end, eps_decay, total_steps):
    eps_decay = (eps_start - eps_end) / total_steps * eps_decay
    def _thunk(steps_done):
        return eps_end + (eps_start - eps_end) * math.exp(-eps_decay * steps_done)
    return _thunk
