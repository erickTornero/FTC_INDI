from abc import abstractclassmethod, abstractmethod
import numpy as np

class BaseController:
    def __init__(self, *args) -> None:
        pass

    @abstractmethod
    def get_action(self, obs: np.ndarray, target_pos: np.ndarray) -> np.ndarray: pass