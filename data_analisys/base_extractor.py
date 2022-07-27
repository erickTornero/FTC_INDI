import os
import json
from wrapper import StateSpaceRobots
class BaseExtractor:
    """
        Stract specific data from the datase to further analisys
        @self.matrix: Matrix variable from the data is obtained, it can be changed
    """
    def __init__(self, folder: os.path):
        self.folder     =   folder
        self.config     =   self.get_config(folder)
        self.names      =   self._get_state_space_names()
        self.indexes    =   self._get_indexes_in_db()
        self._initialize_data()

    def get_positions(self, concat=True):
        return self._get_data('position', concat)


    def get_abs_position(self, concat=True):
        positions = self.get_positions(concat)
        targets   = self.get_targets(concat)
        if concat:
            #return positions + targets
            return positions
        else:
            #return [pos + tar for pos, tar in zip(positions, targets)]
            return [pos for pos, tar in zip(positions, targets)]


    def get_targets(self, concat=True):
        return self._get_data('target', concat)


    def get_actions(self, concat=True):
        return self._get_data('actions', concat)


    def get_angular_velocities(self, concat=True):
        return self._get_data('angular_vel', concat)


    def get_linear_velocities(self, concat=True):
        return self._get_data('linear_vel', concat)


    def get_rotation_matrixes(self, concat=True):
        return self._get_data('rotation_matrix', concat)


    def get_quaternions(self, concat=True):
        return self._get_data('quaternion', concat)


    def get_euler(self, concat=True):
        """
            Return the euler angles:
            In case It is not defined, apply transformations
            from 'rotation_matrix' or 'quaternions'
        """
        if 'euler' in set(self.names):
            return  self._get_data('euler', concat)
        elif 'rotation_matrix' in set(self.names):
            from mbrl.wrapper_ros.utils import euler_from_matrix_batch
            if concat:
                return euler_from_matrix_batch(self.get_rotation_matrixes(concat))
            else:
                return [euler_from_matrix_batch(batch) for batch in self.get_rotation_matrixes(concat)]
        elif 'quaternion' in set(self.names):
            from mbrl.wrapper_ros.utils import euler_from_quaternion_batch
            if concat:
                return euler_from_quaternion_batch(self.get_quaternions(concat))
            else:
                return [euler_from_quaternion_batch(batch) for batch in self.get_quaternions(concat)]

    def _set_matrix(self, newmatrix):
        self.matrix =   newmatrix

    def _get_data(self, name, *argv):
        assert name in self.indexes.keys(), '{} is not in the state space'.format(name)
        i_start     =   self.indexes[name]['start']
        i_end       =   self.indexes[name]['end']
        data        =   self.matrix[:, i_start:i_end]

        return data
    
    def _get_state_space_names(self):
        return self.config['environment']['state_space_names']

    def get_config(self, folder: os.path):
        with open(os.path.join(folder, 'experiment_config.json'), 'r') as fp:
            return json.load(fp)

    def _initialize_data(self):
        raise NotImplementedError()
   

    def _get_indexes_in_db(self):
        state_space_info   =   StateSpaceRobots(self.names)
        nstack      =    1
        indexes     =   state_space_info.get_state_space_indexes()
        state_size  =   state_space_info.get_state_space_shape()[0]   
        bias        =   (nstack - 1) * state_size

        for key in indexes.keys():
            indexes[key]['start']   +=  bias
            indexes[key]['end']     +=  bias
        
        #TODO: Hardcoded action pos in matrix
        indexes['actions']  = { 'start':    state_size * nstack - 4, 
                                'end':      state_size * nstack
                            }  

        return indexes