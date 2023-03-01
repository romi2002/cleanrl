import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PIDEnv(gym.Env):
    def __init__(self):
        self.signal_len = 100
        self.setpoint = 1
        self.error_tolerance = 0
        self.observation_space = spaces.Dict(
            {
                "signal_history": spaces.Box(-10, 10, shape=(self.signal_len,), dtype=np.float32),
                "kP": spaces.Box(-10, 10, shape=(1,), dtype=np.float32)
            }
        )

        self.state = 0
        self.action_space = spaces.Box(-10, 10, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.state = 0
        return {"signal_history", np.zeros(self.signal_len,), "kP", 0}, {}

    def _system(self, u):
        return u * 0.1

    def step(self, action):
        state = 0
        setpoint = 5
        signal_out = np.zeros(self.signal_len,)
        error = np.zeros(self.signal_len,)

        for i in self.signal_len:
            p = (state - setpoint) * action[0]
            error[i] = state - setpoint
            state = self._system(p)
            signal_out[i] = state

        avg_error = np.abs(np.average(error))

        reward = avg_error / 10

        done = avg_error < self.error_tolerance

        obs = {
            "signal_history": signal_out,
            "kP": action[0]
        }

        return obs, reward, done, False, {}