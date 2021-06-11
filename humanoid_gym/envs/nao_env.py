import os
import gym
from gym import spaces
import pybullet as p
import numpy as np
from qibullet import SimulationManager
from qibullet import NaoVirtual
import time

class NaoEnv(gym.Env):
    """docstring for NaoEnv"""
    def __init__(self):
        super(NaoEnv, self).__init__()
        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=True, auto_step=False)
        self.simulation_manager.setLightPosition(self.client, [0,0,100])
        self.robot = self.simulation_manager.spawnNao(self.client, spawn_ground_plane=True)
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
        self.link_names = []
        for name, link in self.robot.link_dict.items():
            if "Finger" not in name and "Thumb" not in name and 'frame' not in name:
                self.link_names.append(name)
        self.action_space = spaces.Box(np.array(self.lower_limits), np.array(self.upper_limits))
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=self._get_obs().shape, dtype="float32")
        self._max_episode_steps = 1000  # float('inf')

    def _get_obs(self):
        link_translations = []
        link_quaternions = []
        for name in self.link_names:
            translation, quaternion = self.robot.getLinkPosition(name)
            link_translations.append(translation)
            link_quaternions.append(quaternion)
        link_translations = np.concatenate(link_translations, axis=0)
        link_quaternions = np.concatenate(link_quaternions, axis=0)
        obs = np.concatenate([np.array(self.robot.getPosition())/10.0,
                              np.array(self.robot.getAnglesPosition(self.joint_names))/np.pi,
                              np.array(self.robot.getAnglesVelocity(self.joint_names))/10.0,
                              link_translations, link_quaternions])
        return obs

    def step(self, actions):
        pos_before = self.robot.getPosition()

        actions = np.array(self.robot.getAnglesPosition(self.joint_names)) + np.array(actions)
        # set joint angles
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        
        self.robot.setAngles(self.joint_names, actions, 1.0)
        self.simulation_manager.stepSimulation(self.client)

        pos_after = self.robot.getPosition()
        alive_bonus = 5.0
        lin_vel_cost = 1.25 * np.linalg.norm(np.array(pos_after[:2]) - np.array(pos_before[:2]))
        quad_ctrl_cost = 0.1 * np.square(np.array(actions)).sum()
        quad_impact_cost = 0  # .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        torso_height = self.robot.getLinkPosition("torso")[0][2]
        done = torso_height < 0.28 or torso_height > 0.4
        info = {}
        # print(self._get_obs())
        return self._get_obs(), reward, done, info

    def reset(self):
        p.resetBasePositionAndOrientation(self.robot.getRobotModel(), [0, 0, 0.34], [0, 0, 0, 1])
        for joint_name, init_angle in zip(self.joint_names, self.init_angles):
            p.resetJointState(self.robot.getRobotModel(), self.robot.joint_dict[joint_name].getIndex(), init_angle)
        return self._get_obs()

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
