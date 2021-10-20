import os
import gym
from gym import spaces
import pybullet as p
import numpy as np
from qibullet import SimulationManager
from qibullet import NaoVirtual
from qibullet.robot_posture import NaoPosture
import time
import h5py
import random
from scipy.spatial.transform import Rotation as R


class NaoEnv(gym.Env):
    """docstring for NaoEnv"""
    def __init__(self):
        super(NaoEnv, self).__init__()
        # read imitation results
        file = 'inference.h5'
        hf = h5py.File(file, 'r')
        group1 = hf.get('group1')
        self.joint_angles = group1.get('joint_angle')[4:-65:2, 1:]
        self.joint_pos = group1.get('joint_pos')[4:-65:2]
        self.total_frames = self.joint_angles.shape[0]
        self.t = 0

        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=True, auto_step=False)
        self.simulation_manager.setLightPosition(self.client, [0,0,100])
        self.robot = self.simulation_manager.spawnNao(self.client, spawn_ground_plane=True)
        p.setTimeStep(1./180.)

        # stand pose parameters
        pose = NaoPosture('Stand')
        pose_dict = {}
        for joint_name, joint_value in zip(pose.joint_names, pose.joint_values):
            pose_dict[joint_name] = joint_value

        # joint parameters
        self.joint_names = ['HeadYaw', 'HeadPitch', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
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

        # default dynamics properties
        self.joint_dampings = {}  # all zeros
        for i in range(p.getNumJoints(self.robot.getRobotModel())):
            joint_info = p.getJointInfo(self.robot.getRobotModel(), i)
            link_name = joint_info[12].decode("utf-8")
            joint_damping = joint_info[6]
            self.joint_dampings[link_name] = joint_damping
        self.link_mass = {}
        self.mass_center = {}
        for name, link in self.robot.link_dict.items():
            dynamics_info = p.getDynamicsInfo(self.robot.getRobotModel(), link.getIndex())
            self.link_mass[name] = dynamics_info[0]
            self.mass_center[name] = dynamics_info[3]

        # self.action_space = spaces.Box(np.array(self.lower_limits), np.array(self.upper_limits))
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(len(self.joint_names),), dtype="float32")
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(len(self._get_obs())*3,), dtype="float32")
        self.obs_history = []
        self._max_episode_steps = 1000  # float('inf')

    def _get_obs(self):
        # get root transform matrix
        root_translation, root_quaternion = self.robot.getLinkPosition("torso")
        root_transform = np.eye(4)
        root_transform[:3, :3] = R.from_quat(root_quaternion).as_matrix()
        root_transform[:3, 3] = root_translation
        # get local position & rotation
        link_translations = []
        link_quaternions = []
        for name in self.link_names:
            translation, quaternion = self.robot.getLinkPosition(name)
            transform = np.eye(4)
            transform[:3, :3] = R.from_quat(quaternion).as_matrix()
            transform[:3, 3] = translation
            transform = np.linalg.inv(root_transform) @ transform
            translation = transform[:3, 3]
            quaternion = R.from_matrix(transform[:3, :3]).as_quat()
            link_translations.append(translation)
            link_quaternions.append(quaternion)

        link_translations = np.concatenate(link_translations, axis=0)
        link_quaternions = np.concatenate(link_quaternions, axis=0)

        l_sole_pos, _ = self.robot.getLinkPosition("l_sole")
        r_sole_pos, _ = self.robot.getLinkPosition("r_sole")
        l_touch_ground = np.array([l_sole_pos[2] < 0.01], dtype=int)
        r_touch_ground = np.array([r_sole_pos[2] < 0.01], dtype=int)
        l_foot_fsr = self.robot.getTotalFsrValues(["LFsrFL_frame", "LFsrFR_frame", "LFsrRL_frame", "LFsrRR_frame"])
        r_foot_fsr = self.robot.getTotalFsrValues(["RFsrFL_frame", "RFsrFR_frame", "RFsrRL_frame", "RFsrRR_frame"])
        # print(l_foot_fsr, r_foot_fsr)
        obs = np.concatenate([#np.array(self.robot.getPosition())/10.0,
                              R.from_quat(root_quaternion).as_euler('xyz'),
                              np.array(self.robot.getAnglesPosition(self.joint_names))/np.pi,
                              np.array(self.robot.getAnglesVelocity(self.joint_names))/10.0,
                              #link_translations, link_quaternions,
                              l_touch_ground, r_touch_ground,
                              #np.array([l_foot_fsr]), np.array([r_foot_fsr]),
                              #self.joint_angles[self.t].flatten()])
                              np.array([self.t/self.total_frames])])
        return obs

    def _get_obs_history(self):
        self.obs_history.append(self._get_obs())
        if len(self.obs_history) < 3:
            concat_obs = np.concatenate([self.obs_history[-1]]*3, axis=0)
        else:
            concat_obs = np.concatenate(self.obs_history[-3:], axis=0)
        return concat_obs

    def step(self, actions):
        pos_before = self.robot.getPosition()

        actions = np.array(self.joint_angles[self.t]) + np.array(actions) + np.clip(0.1*np.random.randn(len(self.joint_names)), -0.1, 0.1)

        # clipping
        current_angles = []
        for joint_name in self.joint_names:
            joint_state = p.getJointState(self.robot.getRobotModel(), self.robot.joint_dict[joint_name].getIndex())
            current_angles.append(joint_state[0])
        current_angles = np.array(current_angles)
        actions = np.clip(actions, a_min=current_angles-0.3, a_max=current_angles+0.3)

        # set joint angles
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        
        self.robot.setAngles(self.joint_names, actions, 1.0)
        self.simulation_manager.stepSimulation(self.client)

        pos_after = self.robot.getPosition()
        alive_bonus = 5.0

        lin_vel_cost = 4 * 125 * (pos_after[0] - pos_before[0])
        quad_ctrl_cost = 0.1 * np.square(np.array(actions)).sum()
        quad_impact_cost = 0  # .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        torso_height = self.robot.getLinkPosition("torso")[0][2]
        done = torso_height < 0.28 or torso_height > 0.4
        info = {'alive_bonus': alive_bonus, 'lin_vel_cost': lin_vel_cost,
            'quad_ctrl_cost': quad_ctrl_cost, 'quad_impact_cost': quad_impact_cost,
            'alive_bonus': alive_bonus, 'action': np.array(self.joint_angles[self.t]),
            'delta_action': np.array(actions) - np.array(self.joint_angles[self.t])}
        # print(self._get_obs())
        self.t += 1
        if self.t == self.total_frames:
            self.t = 0
        return self._get_obs_history(), reward, done, info

    def reset(self):
        p.resetBasePositionAndOrientation(self.robot.getRobotModel(), [0, 0, 0.34], [0, 0, 0, 1])
        p.resetBaseVelocity(self.robot.getRobotModel(), [0, 0, 0], [0, 0, 0])
        for joint_name, init_angle in zip(self.joint_names, self.init_angles):
            p.resetJointState(self.robot.getRobotModel(), self.robot.joint_dict[joint_name].getIndex(), init_angle, 0)

        # dynamics randomization
        p.setGravity(0, 0, -10 + random.uniform(-1, 1))
        p.changeDynamics(self.simulation_manager.ground_plane, -1, lateralFriction=random.uniform(0.5, 2.0))
        for name, link in self.robot.link_dict.items():
            p.changeDynamics(self.robot.getRobotModel(), link.getIndex(), mass=self.link_mass[name]*random.uniform(0.75, 1.15))

        self.t = 0
        self.obs_history = []
        return self._get_obs_history()

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