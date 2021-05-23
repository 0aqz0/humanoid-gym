import gym, humanoid_gym
import pybullet as p

env = gym.make('dancer-v0')

while True:
    env.render()

    actions = env.action_space.sample()

    observation, reward, done, info = env.step(actions)
