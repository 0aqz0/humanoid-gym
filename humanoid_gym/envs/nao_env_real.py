import os
import gym
from gym import spaces
import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R
from qibullet.robot_posture import NaoPosture
import qi
import time
import threading


class NaoEnvReal(gym.Env):
    """docstring for NaoEnvReal"""
    def __init__(self):
        super(NaoEnvReal, self).__init__()
        self.session = qi.Session()
        self.robot_url = '169.254.204.242'
        self.session.connect(self.robot_url)
        self.motion = self.session.service("ALMotion")
        self.motion.setStiffnesses('Body', 1)
        self.memory = self.session.service("ALMemory")
        self.posture = self.session.service("ALRobotPosture")
        self.posture.goToPosture('Stand', 1)

        # joint parameters
        minAngle = {}
        maxAngle = {}
        limits = self.motion.getLimits("Body")
        jointNames = self.motion.getBodyNames("Body")
        for name, limit in zip(jointNames, limits):
            minAngle[name] = limit[0]
            maxAngle[name] = limit[1]
        self.joint_names = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw',
                            'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
        self.lower_limits = [minAngle[name] for name in self.joint_names]
        self.upper_limits = [maxAngle[name] for name in self.joint_names]

        # stand pose parameters
        pose = NaoPosture('Stand')
        pose_dict = {}
        for joint_name, joint_value in zip(pose.joint_names, pose.joint_values):
            pose_dict[joint_name] = joint_value
        self.init_angles = []
        for joint_name in self.joint_names:
            self.init_angles.append(pose_dict[joint_name])

        # self.action_space = spaces.Box(np.array(self.lower_limits), np.array(self.upper_limits))
        self.obs_history = []
        self.obs_length = 10
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(len(self.joint_names),), dtype="float32")
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(len(self._get_obs())*self.obs_length,), dtype="float32")
        self._max_episode_steps = 1000

    def _get_obs(self):
        # torso rpy
        torsoAngleX = self.memory.getData(
            "Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value")
        torsoAngleY = self.memory.getData(
            "Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value")
        torsoAngleZ = self.memory.getData(
            "Device/SubDeviceList/InertialSensor/AngleZ/Sensor/Value")
        # angles
        angles = np.array(self.motion.getAngles(self.joint_names, True))
        # get foot contact
        l_touch_ground = self.memory.getData('Device/SubDeviceList/LFoot/FSR/RearLeft/Sensor/Value') > 0.1 \
            or self.memory.getData('Device/SubDeviceList/LFoot/FSR/RearRight/Sensor/Value') > 0.1
        r_touch_ground = self.memory.getData(
            'Device/SubDeviceList/RFoot/FSR/RearLeft/Sensor/Value') > 0.1
        # observation
        obs = angles
        return obs

    # def _get_obs_history(self):
    #     self.obs_history.append(self._get_obs())
    #     if len(self.obs_history) < 3:
    #         concat_obs = np.concatenate([self.obs_history[-1]]*3, axis=0)
    #     else:
    #         concat_obs = np.concatenate(self.obs_history[-3:], axis=0)
    #     return concat_obs

    def step(self, actions, joints=None):
        # set joint angles
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()

        self.motion.setAngles(self.joint_names if joints is None else joints, actions, 1.0)

        reward = 0
        done = False
        info = None

        return self._get_obs(), reward, done, info

    def reset(self):
        return self._get_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        pass
