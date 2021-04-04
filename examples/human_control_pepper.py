import gym, humanoid_gym
import pybullet as p

env = gym.make('pepper-v0')

motorIds = []
# robot speeds
for v in ['vx', 'vy', 'vw']:
    motorIds.append(p.addUserDebugParameter(v, -1, 1, 0))
# robot joints
for name, lower, upper, init in zip(env.joint_names, env.lower_limits, env.upper_limits, env.init_angles):
    motorIds.append(p.addUserDebugParameter(name, lower, upper, init))

while True:
    env.render()

    actions = []
    for motorId in motorIds:
        actions.append(p.readUserDebugParameter(motorId))

    observation, reward, done, info = env.step(actions)
