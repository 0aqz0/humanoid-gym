from gym.envs.registration import register

# Pepper
register(
    id='pepper-v0',
    entry_point='humanoid_gym.envs:PepperEnv',
)

# NAO
register(
    id='nao-v0',
    entry_point='humanoid_gym.envs:NaoEnv',
)

# Dancer
register(
    id='dancer-v0',
    entry_point='humanoid_gym.envs:DancerEnv',
)