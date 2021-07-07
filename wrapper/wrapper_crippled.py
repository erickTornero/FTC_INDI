import numpy as np
from .wrapper_ros import WrapperROSQuad

#import torch
class QuadrotorEnvRos(WrapperROSQuad):
    """
    This Class wrapps the multiplicative fault in quadrotors
    We use self.masks to symbolic the fault this is multiplied to the
    real action in self.step function.
    """
    def __init__(self, target_pos, crippled_degree, state_space_names, rate=20, model_name='hummingbird', **kwargs_init):
        
        #state_space_names=['rotation_matrix', 'position', 'linear_vel', 'angular_vel']

        super(QuadrotorEnvRos, self).__init__(model_name, target_pos, rate, state_space_names, **kwargs_init)
        #num_models is a variable that content the num of robots in VectWrapperROSQuad
        crippled_degree     =   crippled_degree if crippled_degree is not None else np.ones(4, dtype=np.float32)
        self.set_task(crippled_degree)
        print('QuadrotorEnvRos class initialized \n{}'.format(self.info()))

    
    def step(self, action):
        """
        Overwrite step function, crip action
        """
        
        faulted_actions = self.mask * action

        return super(QuadrotorEnvRos, self).step(faulted_actions)
        

    def info(self):
        
        info_env = 'Info crippled envs'
        info_env += '\n{} with mask \t->\t{}'.format(self.model_name, list(self.mask))

        return info_env

    def set_targetpos(self, target):
        self.target_pos = target

    
    def set_task(self, crippled_degree):
        assert crippled_degree.shape[0] == 4, 'Shape mask must be 4 (for 4 rotors in a quadrotor): shape mask-> {}'.format(crippled_degree.shape)
        assert np.prod((crippled_degree >=0.0) * (crippled_degree <= 1.0) ), 'Values of crippled mask, must be in range [0.0-1.0]: Check mask-> {}'.format(crippled_degree)
        self.mask          =   crippled_degree

    def get_current_task(self):
        return self.mask

    def sample_task(self):
        """ 
            Return a sample of the crippled quadrotor
        """
        return np.random.uniform(0.0, 1.0, self.action_space.shape)