import gym, humanoid_gym
import pybullet as p

env = gym.make('pepper-v0')
env.render()
observation = env.reset()

motorIds = []
for name, lower, upper, init in zip(env.joint_names, env.lower_limits, env.upper_limits, env.init_angles):
    motorIds.append(p.addUserDebugParameter(name, lower, upper, init))

while True:
    env.render()

    actions = []
    for motorId in motorIds:
        actions.append(p.readUserDebugParameter(motorId))

    observation, reward, done, info = env.step(actions)
