from gym.envs.registration import register

register(
    id='nim-v0',
    entry_point='gym_nim.envs:NimEnv',
)
