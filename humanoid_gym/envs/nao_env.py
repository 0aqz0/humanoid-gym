import os
import gym
from gym import spaces
import pybullet as p
import numpy as np
from qibullet import SimulationManager
from qibullet import NaoVirtual
import time
import h5py
from scipy.spatial.transform import Rotation as R

class NaoEnv(gym.Env):
    """docstring for NaoEnv"""
    def __init__(self, gui=True):
        super(NaoEnv, self).__init__()
        # read imitation results
        file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../inference.h5'))
        hf = h5py.File(file, 'r')
        group1 = hf.get('group1')
        self.joint_angles = group1.get('joint_angle')[4:-65, 1:]
        self.total_frames = self.joint_angles.shape[0]
        self.t = 0

        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=gui, auto_step=False)
        self.simulation_manager.setLightPosition(self.client, [0,0,100])
        self.robot = self.simulation_manager.spawnNao(self.client, spawn_ground_plane=True)

        # change friction
        dynamics_info = p.getDynamicsInfo(self.robot.getRobotModel(), self.robot.link_dict['l_sole'].getIndex())
        print('frictions', dynamics_info[1], dynamics_info[6], dynamics_info[7])
        dynamics_info = p.getDynamicsInfo(self.robot.getRobotModel(), self.robot.link_dict['r_sole'].getIndex())
        print('frictions', dynamics_info[1], dynamics_info[6], dynamics_info[7])
        self.friction = 1.0
        p.changeDynamics(self.robot.getRobotModel(), self.robot.link_dict['l_sole'].getIndex(), lateralFriction=self.friction)
        p.changeDynamics(self.robot.getRobotModel(), self.robot.link_dict['r_sole'].getIndex(), lateralFriction=self.friction)

        self.joint_names = ['HeadYaw', 'HeadPitch', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
        self.lower_limits = []
        self.upper_limits = []
        self.init_angles = []
        for joint_name in self.joint_names:
            joint = self.robot.joint_dict[joint_name]
            self.lower_limits.append(joint.getLowerLimit())
            self.upper_limits.append(joint.getUpperLimit())
            self.init_angles.append(self.robot.getAnglesPosition(joint_name))

        # self.action_space = spaces.Box(np.array(self.lower_limits), np.array(self.upper_limits))
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(len(self.joint_names),), dtype="float32")
        self.obs_history = []
        self.obs_length = 3
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(len(self._get_obs())*self.obs_length,), dtype="float32")
        self._max_episode_steps = 1000

    def _get_obs(self):
        # get root transform matrix
        _, root_quaternion = self.robot.getLinkPosition("torso")

        l_sole_pos, _ = self.robot.getLinkPosition("l_sole")
        r_sole_pos, _ = self.robot.getLinkPosition("r_sole")
        l_touch_ground = np.array([l_sole_pos[2] < 0.01], dtype=int)
        r_touch_ground = np.array([r_sole_pos[2] < 0.01], dtype=int)
        l_foot_fsr = self.robot.getTotalFsrValues(["LFsrFL_frame", "LFsrFR_frame", "LFsrRL_frame", "LFsrRR_frame"])
        r_foot_fsr = self.robot.getTotalFsrValues(["RFsrFL_frame", "RFsrFR_frame", "RFsrRL_frame", "RFsrRR_frame"])
        fsr_values = self.robot.getFsrValues(["LFsrFL_frame", "LFsrFR_frame", "LFsrRL_frame", "LFsrRR_frame",
                                              "RFsrFL_frame", "RFsrFR_frame", "RFsrRL_frame", "RFsrRR_frame"])
        fsr_values = (np.array(fsr_values) == 0)
        # print(l_foot_fsr, r_foot_fsr)
        obs = np.concatenate([root_quaternion,
                              np.array(self.robot.getAnglesPosition(self.joint_names)),
                              np.array(self.robot.getAnglesVelocity(self.joint_names)),
                              fsr_values,
                              self.joint_angles[self.t].flatten()])
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

        actions = np.array(self.joint_angles[self.t]) + np.array(actions)
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
                'alive_bonus': alive_bonus}
        self.t += 1
        if self.t == self.total_frames:
            self.t = 0
        return self._get_obs_history(), reward, done, info

    def reset(self):
        p.resetBasePositionAndOrientation(self.robot.getRobotModel(), [0, 0, 0.34], [0, 0, 0, 1])
        p.resetBaseVelocity(self.robot.getRobotModel(), [0, 0, 0], [0, 0, 0])
        for joint_name, init_angle in zip(self.joint_names, self.init_angles):
            p.resetJointState(self.robot.getRobotModel(), self.robot.joint_dict[joint_name].getIndex(), init_angle, 0)
        self.t = 0
        self.obs_history = []
        return self._get_obs_history()

    def close(self):
        p.disconnect()