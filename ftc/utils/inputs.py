from typing import Union, List
import numpy as np

class Inputs:
    def __init__(self):
        self.x_target = None
        self.y_target = None
        self.z_target = None
        self.yaw_target = None

    def update(self, observation):
        pass

    def update_xtarget(self, x_target: float):
        self.x_target = x_target

    def update_ytarget(self, y_target: float):
        self.y_target = y_target

    def update_ztarget(self, z_target):
        self.z_target = z_target

    def update_position_target(self, position: Union[np.ndarray, List[float]]):
        self.update_xtarget(position[0])
        self.update_ytarget(position[1])
        self.update_ztarget(position[2])

    def update_yaw_target(self, yaw_target):
        self.yaw_target = yaw_target

    @property
    def position_target(self) -> np.ndarray:
        return np.array([
                self.x_target,
                self.y_target,
                self.z_target
            ], dtype=np.float32
        )