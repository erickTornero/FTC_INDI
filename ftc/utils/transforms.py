import numpy as np
def pos_invert_yz(position: np.ndarray) -> np.ndarray:
    newpos = position.copy()
    newpos[1:] = - newpos[1:]
    return newpos