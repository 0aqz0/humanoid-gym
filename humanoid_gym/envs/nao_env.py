import os
import gym
from gym import spaces
import pybullet as p
import numpy as np
from qibullet import SimulationManager
from qibullet import NaoVirtual
import time
import h5py
import random
from scipy.spatial.transform import Rotation as R

def linear_interpolate(data):
    total_frames = data.shape[0]
    new_data = []
    for t in range(2*total_frames-1):
        if t % 2 == 0:
            new_data.append(data[t//2])
        else:
            new_data.append((data[t//2] + data[t//2])/2)
    return np.stack(new_data, axis=0)

class NaoEnv(gym.Env):
    """docstring for NaoEnv"""
    def __init__(self):
        super(NaoEnv, self).__init__()
        # read imitation results
        file = 'inference.h5'
        hf = h5py.File(file, 'r')
        group1 = hf.get('group1')
        self.joint_angles = group1.get('joint_angle')[4:-65, 1:]
        # self.joint_angles = linear_interpolate(self.joint_angles)
        self.joint_pos = group1.get('joint_pos')[4:-65]
        # self.joint_pos = linear_interpolate(self.joint_pos)
        self.total_frames = self.joint_angles.shape[0]
        self.t = 0

        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=True, auto_step=False)
        self.simulation_manager.setLightPosition(self.client, [0,0,100])
        self.robot = self.simulation_manager.spawnNao(self.client, spawn_ground_plane=True)

        # change friction
        # dynamics_info = p.getDynamicsInfo(self.robot.getRobotModel(), self.robot.link_dict['l_sole'].getIndex())
        # print('frictions', dynamics_info[1], dynamics_info[6], dynamics_info[7])
        # dynamics_info = p.getDynamicsInfo(self.robot.getRobotModel(), self.robot.link_dict['r_sole'].getIndex())
        # print('frictions', dynamics_info[1], dynamics_info[6], dynamics_info[7])
        # self.friction = 1.0
        # p.changeDynamics(self.robot.getRobotModel(), self.robot.link_dict['l_sole'].getIndex(), lateralFriction=self.friction)
        # p.changeDynamics(self.robot.getRobotModel(), self.robot.link_dict['r_sole'].getIndex(), lateralFriction=self.friction)
        # p.changeVisualShape(self.robot.getRobotModel(), self.robot.link_dict['l_ankle'].getIndex(), rgbaColor=(255,0,0,1))

        self.joint_names = ['HeadYaw', 'HeadPitch', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
        self.lower_limits = []
        self.upper_limits = []
        self.init_angles = []
        for joint_name in self.joint_names:
            joint = self.robot.joint_dict[joint_name]
            self.lower_limits.append(joint.getLowerLimit())
            self.upper_limits.append(joint.getUpperLimit())
            self.init_angles.append(self.robot.getAnglesPosition(joint_name))
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
                              root_quaternion,
                              np.array(self.robot.getAnglesPosition(self.joint_names))/np.pi,
                              np.array(self.robot.getAnglesVelocity(self.joint_names))/10.0,
                              #link_translations, link_quaternions,
                              l_touch_ground, r_touch_ground,
                              #np.array([l_foot_fsr]), np.array([r_foot_fsr]),
                              self.joint_angles[self.t].flatten()])
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

        actions = np.array(self.joint_angles[self.t]) + np.array(actions)
        # set joint angles
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        
        self.robot.setAngles(self.joint_names, actions, 1.0)
        self.simulation_manager.stepSimulation(self.client)

        pos_after = self.robot.getPosition()
        alive_bonus = 5.0
        # trajectory tracking reward
        # link_translations = []
        # link_quaternions = []
        # for name in self.link_names:
        #     translation, quaternion = self.robot.getLinkPosition(name)
        #     link_translations.append(translation)
        #     link_quaternions.append(quaternion)
        # link_translations = np.stack(link_translations, axis=0)
        # link_translations -= np.array(self.robot.getLinkPosition("torso")[0])
        # link_quaternions = np.stack(link_quaternions, axis=0)
        # t = 0
        # pose_cost = np.square(np.linalg.norm(link_translations - joint_pos[t, 1:], axis=1)).sum()
        # current_angles = []
        # for joint_name in self.joint_names:
        #     current_angles.append(self.robot.getAnglesPosition(joint_name))
        # pose_cost = 0 # 10*((self.joint_angles[self.t] - np.array(actions))**2).mean()

        # row pitch tracking reward
        _, root_quaternion = self.robot.getLinkPosition("torso")
        root_rpy = R.from_quat(root_quaternion).as_euler("xyz", degrees=True)
        hip_rpy = [180, 0, 0]  # self.hip_rpy[self.t]
        rp_tracking_cost = 0.001 * np.square(np.array(root_rpy[1:]) - np.array(hip_rpy[1:])).sum()

        # foot position tracking reward
        root_translation, root_quaternion = self.robot.getLinkPosition("torso")
        root_transform = np.eye(4)
        root_transform[:3, :3] = R.from_quat(root_quaternion).as_matrix()
        root_transform[:3, 3] = root_translation
        l_ankle_pos, l_ankle_quat = self.robot.getLinkPosition("l_ankle")
        l_transform = np.eye(4)
        l_transform[:3, :3] = R.from_quat(l_ankle_quat).as_matrix()
        l_transform[:3, 3] = l_ankle_pos
        l_transform = np.linalg.inv(root_transform) @ l_transform
        l_translation = l_transform[:3, 3]
        l_reference = self.joint_pos[self.t, 20]
        r_ankle_pos, r_ankle_quat = self.robot.getLinkPosition("r_ankle")
        r_transform = np.eye(4)
        r_transform[:3, :3] = R.from_quat(r_ankle_quat).as_matrix()
        r_transform[:3, 3] = r_ankle_pos
        r_transform = np.linalg.inv(root_transform) @ r_transform
        r_translation = r_transform[:3, 3]
        r_reference = self.joint_pos[self.t, 26]
        # print(l_translation, l_reference, r_translation, r_reference)
        foot_tracking_cost = 1.0 * (np.square(l_translation - l_reference).sum() + np.square(r_translation - r_reference).sum())

        # zmp reward
        from shapely.geometry import Point
        from shapely.geometry.polygon import Polygon
        l_sole_pos, l_sole_quat = self.robot.getLinkPosition("l_sole")
        l_sole_matrix = R.from_quat(l_sole_quat).as_matrix()
        l_p1 = l_sole_pos + l_sole_matrix @ [-0.06, 0.05, 0]
        l_p2 = l_sole_pos + l_sole_matrix @ [0.1, 0.05, 0]
        l_p3 = l_sole_pos + l_sole_matrix @ [-0.06, -0.04, 0]
        l_p4 = l_sole_pos + l_sole_matrix @ [0.1, -0.04, 0]
        # p.addUserDebugLine(l_p1, l_p2)
        # p.addUserDebugLine(l_p3, l_p4)
        # p.addUserDebugLine(l_p1, l_p3)
        # p.addUserDebugLine(l_p2, l_p4)
        l_polygon = Polygon([l_p1[:2], l_p2[:2], l_p3[:2], l_p4[:2]])
        r_sole_pos, r_sole_quat = self.robot.getLinkPosition("r_sole")
        r_sole_matrix = R.from_quat(r_sole_quat).as_matrix()
        r_p1 = r_sole_pos + r_sole_matrix @ [-0.06, -0.05, 0]
        r_p2 = r_sole_pos + r_sole_matrix @ [0.1, -0.05, 0]
        r_p3 = r_sole_pos + r_sole_matrix @ [-0.06, 0.04, 0]
        r_p4 = r_sole_pos + r_sole_matrix @ [0.1, 0.04, 0]
        # p.addUserDebugLine(r_p1, r_p2)
        # p.addUserDebugLine(r_p3, r_p4)
        # p.addUserDebugLine(r_p1, r_p3)
        # p.addUserDebugLine(r_p2, r_p4)
        r_polygon = Polygon([r_p1[:2], r_p2[:2], r_p3[:2], r_p4[:2]])
        m_polygon = Polygon([l_p3[:2], l_p4[:2], r_p3[:2], r_p4[:2]])
        root_translation, _ = self.robot.getLinkPosition("torso")
        root_point = Point(root_translation[:2])
        zmp_cost = 0.5 if l_polygon.contains(root_point) or r_polygon.contains(root_point) or m_polygon.contains(root_point) else 0.0

        lin_vel_cost = 4 * 125 * (pos_after[0] - pos_before[0])
        quad_ctrl_cost = 0.1 * np.square(np.array(actions)).sum()
        quad_impact_cost = 0  # .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        # reward -= rp_tracking_cost
        # reward -= foot_tracking_cost
        # reward += zmp_cost
        torso_height = self.robot.getLinkPosition("torso")[0][2]
        done = torso_height < 0.28 or torso_height > 0.4
        info = {'alive_bonus': alive_bonus, 'rp_tracking_cost': rp_tracking_cost,
                'foot_tracking_cost': foot_tracking_cost, 'zmp_cost': zmp_cost,
                'lin_vel_cost': lin_vel_cost, 'quad_ctrl_cost': quad_ctrl_cost,
                'quad_impact_cost': quad_impact_cost, 'alive_bonus': alive_bonus}
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
