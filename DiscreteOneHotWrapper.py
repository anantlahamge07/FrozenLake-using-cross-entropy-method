import gymnasium as gym
import numpy as np

class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        # asserting whether the given environments's observation space is an instance of Discrete
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.shape = (env.observation_space.n,)
        self._observation_space = gym.spaces.Box(0.0, 1.0, self.shape, dtype=np.float32)

    def observation(self, observation):
        # getting an one dimensional tensor with all the values set to low (0 in our case)
        t = np.copy(self.observation_space.low)
        t[observation] = 1.0
        return t

