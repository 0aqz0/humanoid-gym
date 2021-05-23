import os
import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import pybullet_data
import numpy as np

class DancerEnv(gym.Env):
    """docstring for DancerEnv"""
    def __init__(self):
        super(DancerEnv, self).__init__()
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0,0,0.1])
        self.reset()

    def step(self, action, custom_reward=None):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        p.setJointMotorControlArray(self.dancerUid, [self.joint2Index[joint] for joint in self.joint_names], p.POSITION_CONTROL, action)
        p.stepSimulation()
        # get states
        jointStates = {}
        for joint in self.joint_names:
            jointStates[joint] = p.getJointState(self.dancerUid, self.joint2Index[joint])
        linkStates = {}
        for link in self.link_names:
            linkStates[link] = p.getLinkState(self.dancerUid, self.link2Index[link])
        # recover color
        for index, color in self.linkColor.items():
            p.changeVisualShape(self.dancerUid, index, rgbaColor=color)
        # check collision
        collision = False
        for link in self.link_names:
            if len(p.getContactPoints(bodyA=self.dancerUid, linkIndexA=self.link2Index[link])) > 0:
                collision = True
                for contact in p.getContactPoints(bodyA=self.dancerUid, bodyB=self.dancerUid, linkIndexA=self.link2Index[link]):
                    print("Collision Occurred in Link {} & Link {}!!!".format(contact[3], contact[4]))
                    p.changeVisualShape(self.dancerUid, contact[3], rgbaColor=[1,0,0,1])
                    p.changeVisualShape(self.dancerUid, contact[4], rgbaColor=[1,0,0,1])
        
        self.step_counter += 1

        if custom_reward is None:
            # default reward
            reward = 0
            done = False
        else:
            # custom reward
            reward, done = custom_reward(jointStates=jointStates, linkStates=linkStates, collision=collision, step_counter=self.step_counter)

        info = {'collision': collision}
        observation = [jointStates[joint][0] for joint in self.joint_names]
        return observation, reward, done, info

    def reset(self):
        p.resetSimulation()
        self.step_counter = 0
        self.dancerUid = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),
            "assets/dancer/dancer_urdf_model.URDF"), basePosition=[0.8,-0.5,0.3], baseOrientation=[-0.7071068,0,0,0.7071068],
            flags=p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_id = p.loadMJCF("mjcf/ground_plane.xml") # ground plane
        p.setGravity(0,0,-10)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(1./240.)
        self.joint_names = []
        self.joint2Index = {} # index map to jointName
        self.link_names = []
        self.link2Index = {} # index map to linkName
        self.lower_limits = []
        self.upper_limits = []
        self.init_angles = []
        for index in range(p.getNumJoints(self.dancerUid)):
            jointName = p.getJointInfo(self.dancerUid, index)[1].decode('utf-8')
            linkName = p.getJointInfo(self.dancerUid, index)[12].decode('utf-8')
            self.joint_names.append(jointName)
            self.joint2Index[jointName] = index
            self.link_names.append(linkName)
            self.link2Index[linkName] = index
            self.lower_limits.append(-np.pi)
            self.upper_limits.append(np.pi)
            self.init_angles.append(0)
        # modify initial angles to avoid collision
        self.init_angles[7], self.init_angles[13] = -0.05, 0.05
        self.linkColor = {} # index map to jointColor
        for data in p.getVisualShapeData(self.dancerUid):
            linkIndex, rgbaColor = data[1], data[7]
            self.linkColor[linkIndex] = rgbaColor
        self.action_space = spaces.Box(np.array([-1]*len(self.joint_names)), np.array([1]*len(self.joint_names)))
        self.observation_space = spaces.Box(np.array([-1]*len(self.joint_names)), np.array([1]*len(self.joint_names)))

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
