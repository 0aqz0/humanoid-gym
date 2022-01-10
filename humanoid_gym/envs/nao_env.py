import os
import gym
from gym import spaces
import pybullet as p
import numpy as np
from qibullet import SimulationManager
from qibullet import NaoVirtual
from qibullet.robot_posture import NaoPosture
import time
from scipy.spatial.transform import Rotation as R
import numpy as np

class NaoEnv(gym.Env):
    """docstring for NaoEnv"""
    def __init__(self):
        super(NaoEnv, self).__init__()
        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=True)
        self.simulation_manager.setLightPosition(self.client, [0,0,100])
        self.robot = self.simulation_manager.spawnNao(self.client, spawn_ground_plane=True)
        time.sleep(1.0)

        # stand pose parameters
        pose = NaoPosture('Stand')
        pose_dict = {}
        for joint_name, joint_value in zip(pose.joint_names, pose.joint_values):
            pose_dict[joint_name] = joint_value

        # joint parameters
        self.joint_names = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw',
                            'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
        self.lower_limits = []
        self.upper_limits = []
        self.init_angles = []
        for joint_name in self.joint_names:
            joint = self.robot.joint_dict[joint_name]
            self.lower_limits.append(joint.getLowerLimit())
            self.upper_limits.append(joint.getUpperLimit())
            self.init_angles.append(pose_dict[joint_name])
        self.link_names = []
        for joint_name in self.joint_names:
            linkName = p.getJointInfo(self.robot.getRobotModel(), self.robot.joint_dict[joint_name].getIndex())[12].decode("utf-8")
            self.link_names.append(linkName)

        self.action_space = spaces.Box(np.array(self.lower_limits), np.array(self.upper_limits))
        self.observation_space = spaces.Box(np.array([-1]*len(self.joint_names)), np.array([1]*len(self.joint_names)))

    def step(self, actions, joints=None):
        # r_wrist_pos = p.getLinkState(self.robot.getRobotModel(), self.robot.link_dict['Head'].getIndex())[0]
        # r_wrist_ori = p.getLinkState(self.robot.getRobotModel(), self.robot.link_dict['Head'].getIndex())[1]
        # r_wrist_rot = R.from_quat(r_wrist_ori).as_matrix() @ R.from_euler('zyx', [90, 90, 0], degrees=True).as_matrix()
        # r_wrist_pos = p.getLinkState(self.robot.getRobotModel(), self.robot.link_dict['l_wrist'].getIndex())[0]
        # r_wrist_ori = p.getLinkState(self.robot.getRobotModel(), self.robot.link_dict['l_wrist'].getIndex())[1]
        # r_wrist_rot = R.from_quat(r_wrist_ori).as_matrix() @ R.from_euler('zyx', [180, 90, 0], degrees=True).as_matrix()  # R.from_euler('y', 90, degrees=True).as_matrix() @ R.from_euler('z', 180, degrees=True).as_matrix()
        # r_wrist_pos = p.getLinkState(self.robot.getRobotModel(), self.robot.link_dict['r_wrist'].getIndex())[0]
        # r_wrist_ori = p.getLinkState(self.robot.getRobotModel(), self.robot.link_dict['r_wrist'].getIndex())[1]
        # r_wrist_rot = R.from_quat(r_wrist_ori).as_matrix() @ R.from_euler('zyx', [0, 90, 0], degrees=True).as_matrix()  # R.from_euler('y', 90, degrees=True).as_matrix() @ R.from_euler('z', 180, degrees=True).as_matrix()
        # r_wrist_pos = p.getLinkState(self.robot.getRobotModel(), self.robot.link_dict['l_ankle'].getIndex())[0]
        # r_wrist_ori = p.getLinkState(self.robot.getRobotModel(), self.robot.link_dict['l_ankle'].getIndex())[1]
        # r_wrist_rot = R.from_quat(r_wrist_ori).as_matrix() @ R.from_euler('zyx', [90, 90, 0], degrees=True).as_matrix()  # R.from_euler('y', 90, degrees=True).as_matrix() @ R.from_euler('z', 180, degrees=True).as_matrix()
        # x_axis = r_wrist_pos + r_wrist_rot @ np.array([1, 0, 0])
        # y_axis = r_wrist_pos + r_wrist_rot @ np.array([0, 1, 0])
        # z_axis = r_wrist_pos + r_wrist_rot @ np.array([0, 0, 1])
        # p.addUserDebugLine(r_wrist_pos, x_axis, lineColorRGB=[1,0,0], lineWidth=1, lifeTime=0.2)
        # p.addUserDebugLine(r_wrist_pos, y_axis, lineColorRGB=[0,1,0], lineWidth=1, lifeTime=0.2)
        # p.addUserDebugLine(r_wrist_pos, z_axis, lineColorRGB=[0,0,1], lineWidth=1, lifeTime=0.2)
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        # set joint angles
        self.robot.setAngles(self.joint_names if joints is None else joints, actions, 1.0)

        # get observations
        observation = self.robot.getAnglesPosition(self.joint_names)
        # TODO: add more observations

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
        pass

    def close(self):
        p.disconnect()
