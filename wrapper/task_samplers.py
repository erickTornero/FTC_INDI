import random
import numpy as np
class TaskSampler:
    """
        Task sampler
        @indexes: allowed indexes to be sampled for failures
    """
    def __init__(self, indexes=[], max_damaged_rotors_per_task=0):
        self.indexes = indexes
        self.random_gen = np.random.default_rng(2021)
        # check
        for index in self.indexes:
            if index >= 4:
                raise ValueError('Just indexes < 4 are allowed')
            else:
                print('Indexe to sample Rotor -> {}'.format(index))
        if len(indexes) > 0: 
            if max_damaged_rotors_per_task < 1: print("WARN: You specified rotors but maximum was set to 0")
        if max_damaged_rotors_per_task > 0: 
            if len(indexes) < 1: print('WARN: Max failed rotors are > 0, but you didnt specified indexes')
        
        assert len(indexes) >= max_damaged_rotors_per_task, 'indexes must be greater than maximum number of rotors to fail'
        self.rotors_to_fail = [0] * max_damaged_rotors_per_task
        self.max_damaged_rotors_per_task = max_damaged_rotors_per_task
        

    def sample_task(self):
        """ 
            sample rotors to be damaged, then the degree of failure
            used to be used when len(index) > max_damaged_rotors_per_task per task
        """
        mask = np.ones(4)
        if self.max_damaged_rotors_per_task > 0:
            rotors_to_fail = self.random_gen.choice(self.indexes, self.max_damaged_rotors_per_task) 
            degre_failures = self.random_gen.uniform(0, 1.0, self.max_damaged_rotors_per_task)
            for index, failure in zip(rotors_to_fail, degre_failures):
                mask[index] = failure
        return mask

    def sampler_v2(self):
        mask = np.ones(4)
        degre_failures = np.random.uniform(0.0, 1.0, self.max_damaged_rotors_per_task)
        for index, failure in zip(self.indexes, degre_failures):
            mask[index] = failure
        return mask