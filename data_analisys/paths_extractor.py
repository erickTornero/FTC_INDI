import os
import json
import joblib
import numpy as np
from wrapper import StateSpaceRobots
from data_analisys import BaseExtractor

class PathsExtractor(BaseExtractor):
    """
        Get the paths experiment
        if paths is given: a list of paths, just use those paths,
        otherwise: use all paths in the rolls
        "Index paths must be between [1-len(paths)]
    """
    def __init__(self, folder, roll_idx, index_paths=None, file_name='paths.pkl'):
        self.roll_path      =   os.path.join(folder, roll_idx)
        self.file_path      =   os.path.join(self.roll_path, file_name)
        self.index_paths    =   index_paths
        
        super(PathsExtractor, self).__init__(self.roll_path)


    def _initialize_data(self):
        self.paths       =   self._get_roll_path()
        if self.index_paths is not None:
            assert type(self.index_paths)==list, 'Indexes of paths must be a list'
            self.paths  =   [self.paths[index-1] for index in self.index_paths]

    def _get_indexes_in_db(self):
        indexes =   super(PathsExtractor, self)._get_indexes_in_db()
        # TODO: HARDCODED indexes
        indexes['actions']   =   {'start': -4, 'end': None}
        indexes['target']   =   {'start': None, 'end':None}
        return indexes


    def _get_data(self, name: str, concat: bool, is_in_matrix=False):
        """
            Get partial columns of data regarding the name
            @name: name of feature e.g. euler, position, etc
            @concat: either to concatenate an a single np.ndarray object or a list of features
        """
        assert type(self.paths) == type(list())
        _data   =   []
        for path in self.paths:
            try:
                self.matrix =   path[name]
                _data.append(self.matrix)
            except KeyError:
                self.matrix =   path['observation']
                _data.append(super(PathsExtractor,self)._get_data(name))
        if concat: 
            return np.concatenate(_data, axis=0)
        return _data
    

    def _get_roll_path(self):
        assert os.path.exists(self.file_path), 'File path <{}> do not exists!'.format(self.file_path)
        return joblib.load(self.file_path)


    def get_config_roll(self):
        config_path     =   os.path.join(self.roll_path, 'experiment_config.json')

        with open(config_path, 'r') as fc:
            return json.load(fc)


    def get_cum_reward(self):
        return np.mean([np.sum(path['rewards']) for path in self.paths])


    def _get_row_elements(self, name, concat):
        _local_data    =   []
        for path in self.paths:
            _local_data.append(path[name])
        if concat:
            return np.concatenate(_local_data, axis=0)
        return _local_data


    def get_row_observations(self, concat=False):
        return self._get_row_elements('observation', concat)
    
    
    def get_row_actions(self, concat=False):
        return self._get_row_elements('actions', concat)

    def get_ndes(self, concat=False):
        return self._get_data('ndes', concat)

    def get_yaw_speed_cmd(self, concat=False):
        return self._get_data('yaw_speed_cmd', concat)

if __name__ == "__main__":
    sp = PathsExtractor('data/sample16/offline_training/train_1/', 'rolls1')
    sp._plot_scatters('j')