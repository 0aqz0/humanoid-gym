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
from collections import deque

class NaoEnv(gym.Env):
    """docstring for NaoEnv"""
    def __init__(self, gui=True):
        super(NaoEnv, self).__init__()
        # read imitation results
        file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../inference-new.h5'))
        hf = h5py.File(file, 'r')
        group1 = hf.get('group1')
        self.joint_angles = group1.get('joint_angle')[113:286, -12:]
        self.total_frames = self.joint_angles.shape[0]
        self.t = 0

        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=gui, auto_step=False)
        self.simulation_manager.setLightPosition(self.client, [0,0,100])
        self.robot = self.simulation_manager.spawnNao(self.client, spawn_ground_plane=True)

        # self.joint_names = ['HeadYaw', 'HeadPitch', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
        self.joint_names = ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
        self.lower_limits = []
        self.upper_limits = []
        for joint_name in self.joint_names:
            joint = self.robot.joint_dict[joint_name]
            self.lower_limits.append(joint.getLowerLimit())
            self.upper_limits.append(joint.getUpperLimit())
        self.init_angles = self.joint_angles[0].tolist()

        # default dynamics properties
        self.joint_dampings = {}  # all zeros
        for i in range(p.getNumJoints(self.robot.getRobotModel())):
            joint_info = p.getJointInfo(self.robot.getRobotModel(), i)
            link_name = joint_info[12].decode("utf-8")
            self.joint_dampings[link_name] = joint_info[6]
        self.link_mass = {}
        self.local_inertia = {}
        self.mass_center = {}
        for name, link in self.robot.link_dict.items():
            dynamics_info = p.getDynamicsInfo(self.robot.getRobotModel(), link.getIndex())
            self.link_mass[name] = dynamics_info[0]
            self.local_inertia[name] = dynamics_info[2]
            self.mass_center[name] = dynamics_info[3]
        self.kps = None
        self.kds = None

        # self.action_space = spaces.Box(np.array(self.lower_limits), np.array(self.upper_limits))
        self.action_space = spaces.Box(low=-0.3, high=0.3, shape=(len(self.joint_names),), dtype="float32")
        self.ang_history = deque(maxlen=100)
        for i in range(100):
            self.ang_history.append(self.robot.getAnglesPosition(self.joint_names))
        self.pos_history = deque(maxlen=100)
        for i in range(100):
            self.pos_history.append(np.array(self.robot.getPosition()))
        self.obs_history = deque(maxlen=100)
        for i in range(100):
            self.obs_history.append(self._get_obs())
        self.obs_length = 3
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(len(self._get_obs())*self.obs_length,), dtype="float32")
        self._max_episode_steps = 1000

    def _get_obs(self):
        # get root transform matrix
        root_quaternion = np.array(self.robot.getLinkPosition("torso")[1])
        rpy = R.from_quat(root_quaternion).as_euler('xyz')
        root_quaternion = np.concatenate([rpy, np.cos(rpy), np.sin(rpy)])
        # root_quaternion += np.random.normal(scale=0.1, size=root_quaternion.shape)
        # root_velocity = np.array(p.getLinkState(
        #                     self.robot.getRobotModel(),
        #                     self.robot.imu.imu_link.getIndex(),
        #                     computeLinkVelocity=True)[6])
        root_velocity = 120*(np.array(self.robot.getPosition())-self.pos_history[-1])
        self.pos_history.append(np.array(self.robot.getPosition()))
        # angles
        angles = np.array(self.robot.getAnglesPosition(self.joint_names))
        # angles += np.random.normal(scale=0.1, size=angles.shape)
        # velocities
        # velocities = np.array(self.robot.getAnglesVelocity(self.joint_names))
        velocities = 120*(angles - self.ang_history[-1])
        # velocities += np.random.normal(scale=0.1, size=velocities.shape)
        self.ang_history.append(angles)
        # foot contact
        l_sole_pos, _ = self.robot.getLinkPosition("l_sole")
        r_sole_pos, _ = self.robot.getLinkPosition("r_sole")
        l_touch_ground = np.array([l_sole_pos[2] < 0.01], dtype=int)
        r_touch_ground = np.array([r_sole_pos[2] < 0.01], dtype=int)
        fsr_values = self.robot.getFsrValues(["LFsrFL_frame", "LFsrFR_frame", "LFsrRL_frame", "LFsrRR_frame",
                                              "RFsrFL_frame", "RFsrFR_frame", "RFsrRL_frame", "RFsrRR_frame"])
        fsr_values = (np.array(fsr_values) == 0)
        # phase
        phase = np.array([self.t/self.total_frames])
        # gyroscope values
        gyroscope = np.array(p.getLinkState(
                            self.robot.getRobotModel(),
                            self.robot.imu.imu_link.getIndex(),
                            computeLinkVelocity=True)[7])
        # gyroscope += np.random.normal(scale=0.1, size=gyroscope.shape)
        # observation
        obs = np.concatenate([root_quaternion,
                              root_velocity,
                              gyroscope,
                              angles,
                              #velocities,
                              fsr_values,
                              phase])
        return obs

    def _get_obs_history(self):
        self.obs_history.append(self._get_obs())
        concat_obs = np.concatenate([self.obs_history[-1], self.obs_history[-11], self.obs_history[-21]], axis=0)
        return concat_obs

    def step(self, actions):
        pos_before = self.robot.getPosition()

        actions = np.array(self.joint_angles[self.t]) + np.array(actions)
        # LHipYawPitch equals to RHipYawPitch
        actions[-6] = actions[-12]
        # set joint angles
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        
        self.robot.setAngles(self.joint_names, actions, 0.3, self.kps, self.kds)
        # step twice to 120 Hz
        self.simulation_manager.stepSimulation(self.client)
        self.simulation_manager.stepSimulation(self.client)

        pos_after = self.robot.getPosition()
        alive_bonus = 5.0
        lin_vel_cost = 4 * 125 * (pos_after[0] - pos_before[0])
        quad_ctrl_cost = 0  # 0.1 * np.square(np.array(actions)).sum()
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
        self.t += 1
        if self.t == self.total_frames:
            self.t = 0
        return self._get_obs_history(), reward, done, info

    def reset(self):
        p.resetBasePositionAndOrientation(self.robot.getRobotModel(), [0, 0, 0.33], [0, 0, 0, 1])
        p.resetBaseVelocity(self.robot.getRobotModel(), [0, 0, 0], [0, 0, 0])
        # dynamics randomization
        # change friction
        dynamics_info = p.getDynamicsInfo(self.robot.getRobotModel(), self.robot.link_dict['l_sole'].getIndex())
        dynamics_info = p.getDynamicsInfo(self.robot.getRobotModel(), self.robot.link_dict['r_sole'].getIndex())
        self.foot_friction = random.uniform(0.1, 0.5)
        self.ground_friction = random.uniform(0.5, 3.0)
        p.changeDynamics(self.robot.getRobotModel(), self.robot.link_dict['l_sole'].getIndex(), lateralFriction=self.foot_friction)
        p.changeDynamics(self.robot.getRobotModel(), self.robot.link_dict['r_sole'].getIndex(), lateralFriction=self.foot_friction)
        p.changeDynamics(self.simulation_manager.ground_plane, -1, lateralFriction=self.ground_friction)
        # change gravity
        p.setGravity(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), -10 + random.uniform(-1, 1))
        # change mass & local inertia & joint damping
        for name, link in self.robot.link_dict.items():
            p.changeDynamics(self.robot.getRobotModel(), link.getIndex(),
                             mass=self.link_mass[name]*random.uniform(0.75, 1.15),
                             # localInertiaDiagnoal=np.array(self.local_inertia[name])*random.uniform(0.75, 1.15),
                             jointDamping=self.joint_dampings[name]*random.uniform(0.75, 1.15))
        # change Kp & Kd
        self.kp = [1e-1] * len(self.joint_names)
        self.kd = [8e-1] * len(self.joint_names)

        # stand pose parameters
        pose = NaoPosture('Stand')
        for joint_name, init_angle in zip(pose.joint_names, pose.joint_values):
            p.resetJointState(self.robot.getRobotModel(), self.robot.joint_dict[joint_name].getIndex(), init_angle, 0)
            p.setJointMotorControl2(self.robot.getRobotModel(),
                                    self.robot.joint_dict[joint_name].getIndex(),
                                    p.POSITION_CONTROL,
                                    targetPosition=init_angle)
        for joint_name, init_angle in zip(self.joint_names, self.init_angles):
            p.resetJointState(self.robot.getRobotModel(), self.robot.joint_dict[joint_name].getIndex(), init_angle, 0)
            p.setJointMotorControl2(self.robot.getRobotModel(),
                                    self.robot.joint_dict[joint_name].getIndex(),
                                    p.POSITION_CONTROL,
                                    targetPosition=init_angle)
        for _ in range(400):
            p.stepSimulation()
        self.t = 0
        self.ang_history = deque(maxlen=100)
        for i in range(100):
            self.ang_history.append(self.robot.getAnglesPosition(self.joint_names))
        self.pos_history = deque(maxlen=100)
        for i in range(100):
            self.pos_history.append(np.array(self.robot.getPosition()))
        self.obs_history = deque(maxlen=100)
        for i in range(100):
            self.obs_history.append(self._get_obs())
        return self._get_obs_history()

    def close(self):
        p.disconnect()