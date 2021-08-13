import time
import numpy as np
from math import sin, cos

class LQRController:
    def __init__(self, *argsv):
        raise NotImplemented("")

    def __call__(self, state, inputs):
        """Implement here a forward pass of the controller"""
        w_cmd = np.zeros(4, )
        return w_cmd.flatten()