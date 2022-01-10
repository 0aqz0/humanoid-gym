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

# Romeo
register(
    id='romeo-v0',
    entry_point='humanoid_gym.envs:RomeoEnv',
)

# Dancer
register(
    id='dancer-v0',
    entry_point='humanoid_gym.envs:DancerEnv',
)

# NAO Real Robot
register(
    id='nao-real-v0',
    entry_point='humanoid_gym.envs:NaoEnvReal',
)
