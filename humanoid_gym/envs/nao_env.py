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
        file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../inference.h5'))
        hf = h5py.File(file, 'r')
        group1 = hf.get('group1')
        self.joint_angles = group1.get('joint_angle')[4:-65, 3:]
        self.joint_angles = np.concatenate([self.joint_angles[:, :5], self.joint_angles[:, 6:11], self.joint_angles[:, 12:]], axis=1)
        self.joint_angles[:, -6] = self.joint_angles[:, 10]
        self.joint_angles = self.joint_angles[:, -12:]
        self.total_frames = self.joint_angles.shape[0]
        self.t = 0

        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=True, auto_step=False)
        self.simulation_manager.setLightPosition(self.client, [0,0,100])
        self.robot = self.simulation_manager.spawnNao(self.client, spawn_ground_plane=True)
        p.setTimeStep(1./120.)

        # stand pose parameters
        pose = NaoPosture('Stand')
        pose_dict = {}
        for joint_name, joint_value in zip(pose.joint_names, pose.joint_values):
            pose_dict[joint_name] = joint_value

        # joint parameters
        # self.joint_names = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
        self.joint_names = ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
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
        self.obs_history = []
        self.obs_length = 3
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(len(self.joint_names),), dtype="float32")
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(len(self._get_obs())*self.obs_length,), dtype="float32")
        self._max_episode_steps = 1000  # float('inf')

    def _get_obs(self):
        # get root transform matrix
        _, root_quaternion = self.robot.getLinkPosition("torso")
        # l_sole_pos, _ = self.robot.getLinkPosition("l_sole")
        # r_sole_pos, _ = self.robot.getLinkPosition("r_sole")
        # l_touch_ground = np.array([l_sole_pos[2] < 0.01], dtype=int)
        # r_touch_ground = np.array([r_sole_pos[2] < 0.01], dtype=int)
        # l_foot_fsr = self.robot.getTotalFsrValues(["LFsrFL_frame", "LFsrFR_frame", "LFsrRL_frame", "LFsrRR_frame"])
        # r_foot_fsr = self.robot.getTotalFsrValues(["RFsrFL_frame", "RFsrFR_frame", "RFsrRL_frame", "RFsrRR_frame"])
        # print(l_foot_fsr, r_foot_fsr)
        fsr_values = self.robot.getFsrValues(["LFsrFL_frame", "LFsrFR_frame", "LFsrRL_frame", "LFsrRR_frame",
                                              "RFsrFL_frame", "RFsrFR_frame", "RFsrRL_frame", "RFsrRR_frame"])
        fsr_values = (np.array(fsr_values) == 0)
        angles = np.array(self.robot.getAnglesPosition(self.joint_names)) + np.clip(0.1*np.random.randn(len(self.joint_names)), -0.1, 0.1)
        obs = np.concatenate([#np.array(self.robot.getPosition())/10.0,
                              R.from_quat(root_quaternion).as_euler('xyz'),
                              angles,
                              #l_touch_ground, r_touch_ground,
                              fsr_values,
                              np.array([self.t/self.total_frames])])
        return obs

    def _get_obs_history(self):
        self.obs_history.append(self._get_obs())
        if len(self.obs_history) < 10:
            concat_obs = np.concatenate([self.obs_history[-1]]*self.obs_length, axis=0)
        else:
            concat_obs = np.concatenate([self.obs_history[-1], self.obs_history[-5], self.obs_history[-9]], axis=0)
        return concat_obs

    def step(self, actions):
        pos_before = self.robot.getPosition()

        actions = np.array(self.joint_angles[int(self.t)]) + np.array(actions)

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

        self.robot.setAngles(self.joint_names, actions, 0.2)
        self.simulation_manager.stepSimulation(self.client)

        pos_after = self.robot.getPosition()
        alive_bonus = 5.0

        lin_vel_cost = 4 * 125 * (pos_after[0] - pos_before[0])
        quad_ctrl_cost = 0.1 * np.square(np.array(actions)).sum()
        quad_impact_cost = 0  # .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        torso_height = self.robot.getLinkPosition("torso")[0][2]
        walking_dist = np.linalg.norm(self.robot.getLinkPosition("torso")[0][:2])
        p.addUserDebugText('Walking Distance: {:.2f} m'.format(walking_dist), np.array(self.robot.getLinkPosition("torso")[0]) + np.array([0, 0, 0.5]),
            textColorRGB=[0, 0, 1], textSize=3, lifeTime=0.1)
        # following view
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=self.robot.getLinkPosition("torso")[0])
        done = torso_height < 0.28 or torso_height > 0.4 or walking_dist > 100
        info = {'alive_bonus': alive_bonus, 'lin_vel_cost': lin_vel_cost,
            'quad_ctrl_cost': quad_ctrl_cost, 'quad_impact_cost': quad_impact_cost,
            'alive_bonus': alive_bonus}
        # print(self._get_obs())
        self.t += 0.5
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

    def close(self):
        p.disconnect()