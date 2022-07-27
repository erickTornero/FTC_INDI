import os
import json
from mbrl.datasets import DatasetMBRL as DataSet
from mbrl.data_analisys import BaseExtractor

from mbrl.wrapper_ros import StateSpaceRobots
class DatasetExtractor(BaseExtractor):
    """
        Stract specific data from the datase to further analisys
        @self.matrix: Matrix variable from the data is obtained, it can be changed
    """
    def __init__(self, folder):
        self.ds_folder  =   os.path.join(folder, 'dataset')
        super(DatasetExtractor, self).__init__(folder)


    def _initialize_data(self, seed=42, split_after=False):
        """Initialize data with the training data"""
        dataset                                     =   DataSet(self.ds_folder, seed=seed, split_after=split_after)
        dataset.load_dataset_all()
        self.features_train, self.features_test,_,_ =   dataset.get_data_train()
        self.use_features_train()
   
    def use_features_train(self):
        self._set_matrix(self.features_train)

    def use_features_test(self):
        self._set_matrix(self.features_test)