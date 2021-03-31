import os
import gym
from gym import spaces
import pybullet as p
import numpy as np
from qibullet import SimulationManager
from qibullet import PepperVirtual

class PepperEnv(gym.Env):
    """docstring for PepperEnv"""
    def __init__(self):
        super(PepperEnv, self).__init__()
        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=True)
        self.robot = self.simulation_manager.spawnPepper(self.client, spawn_ground_plane=True)
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
        self.action_space = spaces.Box(np.array(self.lower_limits), np.array(self.upper_limits))
        # self.observation_space = spaces.Box(np.array([-1]*len(self.joints)), np.array([1]*len(self.joints)))

    def step(self, actions):
        # set action
        self.robot.setAngles(self.joint_names, actions, 1.0)

        # TODO: design your reward
        reward = 0
        done = False
        info = {}

        # observation = [jointStates[joint][0] for joint in self.joints]
        # return observation, reward, done, info
        return None, None, None, None

    def reset(self):
        # p.resetSimulation()
        # self.step_counter = 0
        # self.pepperUid = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #     "assets/pepper.urdf"), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        # p.setGravity(0,0,-10)
        # p.setPhysicsEngineParameter(numSolverIterations=150)
        # p.setTimeStep(1./240.)
        # self.joint2Index = {} # jointIndex map to jointName
        # for i in range(p.getNumJoints(self.pepperUid)):
        #     self.joint2Index[p.getJointInfo(self.pepperUid, i)[1].decode('utf-8')] = i
        # self.jointColor = {} # jointName map to jointColor
        # for data in p.getVisualShapeData(self.pepperUid):
        #     self.jointColor[p.getJointInfo(self.pepperUid, data[1])[1].decode('utf-8')] = data[7]
        # # recover color
        # for joint, index in self.joint2Index.items():
        #     if joint in self.jointColor and joint != 'world_joint':
        #         p.changeVisualShape(self.pepperUid, index, rgbaColor=self.jointColor[joint])
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()