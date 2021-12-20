import gym
from gym import spaces
import numpy as np
import h5py
from qibullet.robot_posture import NaoPosture
import qi
import time
import threading
from collections import deque
import os

class NaoEnvReal(gym.Env):
    """docstring for NaoEnvReal"""
    def __init__(self):
        super(NaoEnvReal, self).__init__()
        # read imitation results
        file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../inference.h5'))
        hf = h5py.File(file, 'r')
        group1 = hf.get('group1')
        self.joint_angles = group1.get('joint_angle')[4:-65:2, 3:]
        self.joint_angles = np.concatenate([self.joint_angles[:, :5], self.joint_angles[:, 6:11], self.joint_angles[:, 12:]], axis=1)
        self.joint_angles[:, -6] = self.joint_angles[:, 10]
        self.total_frames = self.joint_angles.shape[0]
        self.t = 0

        self.session = qi.Session()
        self.robot_url = '169.254.245.252'  # '192.168.199.173'
        self.session.connect(self.robot_url)
        self.motion = self.session.service("ALMotion")

        self.memory = self.session.service("ALMemory")
        self.posture = self.session.service("ALRobotPosture")
        self.motion.setStiffnesses('Body', 1)
        self.motion.setFallManagerEnabled(False)
        # self.posture.goToPosture('Stand', 1)

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

        self.motion.setAngles(pose.joint_names, pose.joint_values, 0.1)

        self._module_termination = False
        # sensor thread
        self.obs_history = deque(maxlen=100)
        self.obs_length = 10
        self.state = None
        self.current_angles = None
        self.sensor_process = threading.Thread(target=self._sensorScan)
        self.sensor_process.start()
        # command thread
        self.command = self.init_angles
        self.command_process = threading.Thread(target=self._sendCommand)
        self.command_process.start()
        time.sleep(1.0)

        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(len(self.joint_names),), dtype="float32")
        self.observation_space = spaces.Box(low=-float('inf'), high=float(
            'inf'), shape=self.state.shape, dtype="float32")
        self._max_episode_steps = 1000
        self.time_step = 1./120.
        self.tick = None

    def step(self, actions):
        actions = np.array(self.joint_angles[int(self.t)])#np.array(actions)
        # clipping
        actions = np.clip(actions, a_min=self.current_angles-0.3, a_max=self.current_angles+0.3)

        # set joint angles
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        
        self.command = actions

        if self.tick is None:
            self.tick = time.time()
        else:
            delta_time = time.time() - self.tick
            # print('delta time: ', delta_time)
            sleep_time = self.time_step - delta_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print('OUT OF TIME', 'delta time: ', delta_time)
            self.tick = time.time()

        reward = 0
        done = False
        info = {'alive_bonus': 0, 'lin_vel_cost': 0, 'quad_ctrl_cost': 0,
                'quad_impact_cost': 0, 'alive_bonus': 0}
        self.t += 0.25
        if self.t >= self.total_frames:
            self.t = 0

        return self.state, reward, done, info

    def reset(self):
        return self.state
        
    def _sensorScan(self):
        while not self._module_termination:
            # update observation history
            # torso rpy
            torsoAngleX = self.memory.getData(
                "Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value")
            torsoAngleY = self.memory.getData(
                "Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value")
            torsoAngleZ = self.memory.getData(
                "Device/SubDeviceList/InertialSensor/AngleZ/Sensor/Value")
            # foot contact
            # l_touch_ground = self.memory.getData('leftFootContact')
            # r_touch_ground = self.memory.getData('rightFootContact')
            # l_touch_ground = self.memory.getData('Device/SubDeviceList/LFoot/FSR/RearLeft/Sensor/Value') > 0.1 \
            #     or self.memory.getData('Device/SubDeviceList/LFoot/FSR/RearRight/Sensor/Value') > 0.1
            # r_touch_ground = self.memory.getData(
            #     'Device/SubDeviceList/RFoot/FSR/RearLeft/Sensor/Value') > 0.1
            LFsrFL_frame = self.memory.getData(
                'Device/SubDeviceList/LFoot/FSR/FrontLeft/Sensor/Value') > 0.1
            LFsrFR_frame = self.memory.getData(
                'Device/SubDeviceList/LFoot/FSR/FrontRight/Sensor/Value') > 0.1
            LFsrRL_frame = self.memory.getData(
                'Device/SubDeviceList/LFoot/FSR/RearLeft/Sensor/Value') > 0.1
            LFsrRR_frame = self.memory.getData(
                'Device/SubDeviceList/LFoot/FSR/RearRight/Sensor/Value') > 0.1
            RFsrFL_frame = self.memory.getData(
                'Device/SubDeviceList/RFoot/FSR/FrontLeft/Sensor/Value') > 0.1
            RFsrFR_frame = self.memory.getData(
                'Device/SubDeviceList/RFoot/FSR/FrontRight/Sensor/Value') > 0.1
            RFsrRL_frame = self.memory.getData(
                'Device/SubDeviceList/RFoot/FSR/RearLeft/Sensor/Value') > 0.1
            RFsrRR_frame = self.memory.getData(
                'Device/SubDeviceList/RFoot/FSR/RearRight/Sensor/Value') > 0.1
            fsr_values = np.array([LFsrFL_frame, LFsrFR_frame, LFsrRL_frame, LFsrRR_frame,
                                   RFsrFL_frame, RFsrFR_frame, RFsrRL_frame, RFsrRR_frame])
            # joint angles
            angles = np.array(self.motion.getAngles(self.joint_names, True))
            obs = np.concatenate([
                np.array([torsoAngleX, torsoAngleY, 0]),
                angles,
                fsr_values,
                np.array([self.t/self.total_frames],
                )])
            self.obs_history.append(obs)

            # update current angles
            self.current_angles = angles

            # update state
            if len(self.obs_history) < self.obs_length:
                self.state = np.concatenate(
                    [self.obs_history[-1]]*self.obs_length, axis=0)
            else:
                self.state = np.concatenate(
                    list(self.obs_history)[-self.obs_length:], axis=0)

    def _sendCommand(self):
        while not self._module_termination:
            start = time.time()
            self.motion.setAngles(self.joint_names, self.command, 0.2)
            end = time.time()
            # print('command', end - start)