from typing import Union, List
import numpy as np

class Inputs:
    def __init__(self):
        self.xTarget = None
        self.yTarget = None
        self.zTarget = None
        self.yawTarget = None

    def update(self, observation):
        pass
    
    def update_xtarget(self, x_target: float):
        self.xTarget = x_target
    
    def update_ytarget(self, y_target: float):
        self.yTarget = y_target

    def update_ztarget(self, z_target):
        self.zTarget = z_target
    
    def update_position_target(self, position: Union[np.ndarray, List[float]]):
        self.update_xtarget(position[0])
        self.update_ytarget(position[1])
        self.update_ztarget(position[2])

    def update_yawTarget(self, yaw_target):
        self.yawTarget = yaw_target