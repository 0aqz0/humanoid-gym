import gym, humanoid_gym
import pybullet as p
import h5py
import time

hf = h5py.File('inference.h5', 'r')
group1 = hf.get('group1')
l_joint_angle = group1.get('l_joint_angle')
r_joint_angle = group1.get('r_joint_angle')
# l_hand_angle = group1.get('l_glove_angle_2')
# r_hand_angle = group1.get('r_glove_angle_2')
total_frames = l_joint_angle.shape[0]
print(l_joint_angle.shape)

env = gym.make('nao-v0')
env.render()

while True:
    env.render()

    for t in range(total_frames):
        action = l_joint_angle[t].tolist() + [1] + r_joint_angle[t].tolist() + [1]
        print(t, action)
        observation, reward, done, info = env.step(action)
        time.sleep(0.1)
