import os
import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np

class NaoEnv(gym.Env):
    """docstring for NaoEnv"""
    def __init__(self):
        super(NaoEnv, self).__init__()
        pass