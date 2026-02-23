from typing import Dict
import numpy as np

class StateSpaceRobots:
    def __init__(self, names):
        self._init_info_state_spaces()
        self.names  =   names
        self.indexes = self.get_state_space_indexes()

    
    def _init_info_state_spaces(self):
        self.states_dict =   {
            'rotation_matrix': 9,
            'position': 3,
            'euler': 3,
            'quaternion': 4,
            'linear_vel': 3,
            'angular_vel': 3,
        }

    def get_state_space_shape(self):
        shape   =   0
        for name in self.names:
            try:
                shape += self.states_dict[name]
            except KeyError as error:
                raise Exception('Invalid name <{}> for state space'.format(name))

        return (shape, ) 

    
    def get_state_space_indexes(self) -> Dict[str, Dict[str, int]]:
        indexes =   {}
        index   =   0
        for name in self.names:
            try:
                indexes[name]   =   {
                    'start':    index,
                    'end':      index + self.states_dict[name],
                }
                index += self.states_dict[name] 
            except KeyError as error:
                raise Exception('Invalid name <{}> for state space'.format(name))

        return indexes

    def get_from_vector(self, vector, start, end):
        return vector[start:end]

    def get_values(self, vector):
        return (self.get_from_vector(vector, **self.indexes[name]) for name in self.names)

    def get_attrib(self, vector: np.ndarray, attrib_name: str):
        return self.get_from_vector(vector, **self.indexes[attrib_name])

    def get_obs_dict(self, vector: np.ndarray):
        return {attrib: self.get_attrib(vector, attrib) for attrib in self.names}