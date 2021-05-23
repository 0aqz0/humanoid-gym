import os
import gym
from gym import spaces
import pybullet as p
import numpy as np
from qibullet import SimulationManager
from qibullet import RomeoVirtual
import time

class RomeoEnv(gym.Env):
    """docstring for RomeoEnv"""
    def __init__(self):
        super(RomeoEnv, self).__init__()
        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=True)
        self.simulation_manager.setLightPosition(self.client, [0,0,100])
        self.robot = self.simulation_manager.spawnRomeo(self.client, spawn_ground_plane=True)
        time.sleep(1.0)

        self.joint_names = []
        self.lower_limits = []
        self.upper_limits = []
        self.init_angles = []
        for name, joint in self.robot.joint_dict.items():
            if "Finger" not in name and "Thumb" not in name:
                self.joint_names.append(name)
                self.lower_limits.append(joint.getLowerLimit())
                self.upper_limits.append(joint.getUpperLimit())
                self.init_angles.append(self.robot.getAnglesPosition(name))
        self.action_space = spaces.Box(np.array(self.lower_limits), np.array(self.upper_limits))
        self.observation_space = spaces.Box(np.array([-1]*len(self.joint_names)), np.array([1]*len(self.joint_names)))

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        # set joint angles
        self.robot.setAngles(self.joint_names, actions, 1.0)

        # get observations
        observation = {
            'position': self.robot.getPosition(),
            'anglesPosition': self.robot.getAnglesPosition(self.joint_names),
            'anglesVelocity': self.robot.getAnglesVelocity(self.joint_names),
            # TODO: add more observations
            }

        # TODO: design your reward
        reward = 0
        done = False
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.simulation_manager.removeNao(self.robot)
        self.robot = self.simulation_manager.spawnNao(self.client, spawn_ground_plane=True)
        time.sleep(1.0)

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5,0,0.5],
                                                          distance=.7,
                                                          yaw=90,
                                                          pitch=0,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960)/720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960,4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()
