import timeit
import gymnasium as gym
import gym_usv
import numpy as np

if __name__ == '__main__':
    n_envs = 8
    n_steps = 1000

    sync_vec_env = gym.vector.SyncVectorEnv(
        [lambda: gym.make("usv-asmc-ca-v0")] * n_envs
    )

    asynv_vec_env = gym.vector.AsyncVectorEnv(
        [lambda: gym.make("usv-asmc-ca-v0")] * n_envs
    )

    action = np.zeros((n_envs, 2))

    sync_vec_env.reset()
    asynv_vec_env.reset()

    sync_time = timeit.timeit(lambda: sync_vec_env.step(action), number=n_steps)
    #async_time = timeit.timeit(lambda: asynv_vec_env.step(action), number=n_steps)
    async_time = 0
    
    print(f"Sync time: {sync_time} async time: {async_time}")
