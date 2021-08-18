import os
import gym
from gym import spaces
import pybullet as p
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R
from qibullet.robot_posture import NaoPosture
import qi
import time
import threading


class NaoEnvReal(gym.Env):
    """docstring for NaoEnvReal"""
    def __init__(self):
        super(NaoEnvReal, self).__init__()
        # read imitation results
        file = 'inference.h5'
        hf = h5py.File(file, 'r')
        group1 = hf.get('group1')
        self.joint_angles = group1.get('joint_angle')[4:-65:2, 1:]
        self.joint_pos = group1.get('joint_pos')[4:-65:2]
        self.total_frames = self.joint_angles.shape[0]
        self.t = 0

        self.session = qi.Session()
        self.robot_url = '169.254.136.138'  # '192.168.199.173'
        self.session.connect(self.robot_url)
        self.motion = self.session.service("ALMotion")
        
        self.memory = self.session.service("ALMemory")
        self.posture = self.session.service("ALRobotPosture")
        self.motion.setStiffnesses('Body', 1)
        print(self.motion.getFallManagerEnabled())
        self.motion.setFallManagerEnabled(False)
        self.posture.goToPosture('Stand', 1)

        # joint parameters
        minAngle = {}
        maxAngle = {}
        limits = self.motion.getLimits("Body")
        jointNames = self.motion.getBodyNames("Body")
        for name, limit in zip(jointNames, limits):
            minAngle[name] = limit[0]
            maxAngle[name] = limit[1]
        self.joint_names = ['HeadYaw', 'HeadPitch', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
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

        # velocity thread
        self._module_termination = False
        self.velocity_process = threading.Thread(target=self._velocityScan)
        self.velocity_process.start()

        # self.action_space = spaces.Box(np.array(self.lower_limits), np.array(self.upper_limits))
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(len(self.joint_names),), dtype="float32")
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(len(self._get_obs())*3,), dtype="float32")
        self.obs_history = []
        self._max_episode_steps = 1000  # float('inf')
        self.tick = None

    def _get_obs(self):
        # torso rpy
        torsoAngleX = self.memory.getData("Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value")
        torsoAngleY = self.memory.getData("Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value")
        torsoAngleZ = self.memory.getData("Device/SubDeviceList/InertialSensor/AngleZ/Sensor/Value")
        # print(torsoAngleX, torsoAngleY, torsoAngleZ)
        # get foot contact
        # l_touch_ground = self.memory.getData('leftFootContact')
        # r_touch_ground = self.memory.getData('rightFootContact')
        l_touch_ground = self.memory.getData('Device/SubDeviceList/LFoot/FSR/RearLeft/Sensor/Value') > 0.1 \
                        or self.memory.getData('Device/SubDeviceList/LFoot/FSR/RearRight/Sensor/Value') > 0.1
        r_touch_ground = self.memory.getData('Device/SubDeviceList/RFoot/FSR/RearLeft/Sensor/Value') > 0.1
        # print('angle', self.motion.getAngles(self.joint_names, True))
        # print('speed', self.joint_velocity)
        # print('foot contact', l_touch_ground, r_touch_ground)
        obs = np.concatenate([np.array([torsoAngleX, torsoAngleY, 0]),
                              np.array(self.motion.getAngles(self.joint_names, True))/np.pi,
                              self.joint_velocity/10.0,
                              #np.array(self.robot.getAnglesVelocity(self.joint_names))/10.0,
                              [l_touch_ground], [r_touch_ground],
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
        # print('action:', self.t, actions)
        actions = np.array(self.joint_angles[self.t]) #+ np.array(actions)

        # clipping
        current_angles = np.array(self.motion.getAngles(self.joint_names, True))
        actions = np.clip(actions, a_min=current_angles-0.3, a_max=current_angles+0.3)

        # set joint angles
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        
        # print(input("breakpoint:"))
        # self.motion.setAngles(self.joint_names, actions, 0.5)
        print(self.motion.getFallManagerEnabled())
        # self.motion.changeAngles("HeadYaw", 0.25, 0.05)
        # time.sleep(1.0)

        # set speed to zero
        # time.sleep(1./120.)
        # self.motion.setAngles(self.joint_names, self.motion.getAngles(self.joint_names, True), 1.0)
        if self.tick is None:
            self.tick = time.time()
        else:
            delta_time = time.time() - self.tick
            # print('delta time: ', delta_time)
            sleep_time = 1./100. - delta_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print('OUT OF TIME')
            self.tick = time.time()

        reward = 0
        done = False
        info = {'alive_bonus': 0, 'lin_vel_cost': 0, 'quad_ctrl_cost': 0,
                'quad_impact_cost': 0, 'alive_bonus': 0}

        self.t += 1
        if self.t >= self.total_frames:
            self.t = 0
        return self._get_obs_history(), reward, done, info

    def reset(self):
        return self._get_obs_history()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _velocityScan(self):
        last_joint_angles = np.array(self.motion.getAngles(self.joint_names, True))
        last_time = time.time()
        self.joint_velocity = np.zeros(len(self.joint_names))

        while not self._module_termination:
            time.sleep(0.05)
            current_joint_angles = np.array(self.motion.getAngles(self.joint_names, True))
            current_time = time.time()
            self.joint_velocity = (current_joint_angles - last_joint_angles)/(current_time - last_time)
            # print('speed', self.joint_velocity, current_time - last_time)
            last_joint_angles = current_joint_angles
            last_time = current_time
