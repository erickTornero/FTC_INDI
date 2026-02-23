# Wrapper for the ros & gazebo simulator
# based on the common wrapper
from typing import Dict, Optional, Tuple
import gym
import numpy as np
import rospy

#from msg import rewardinfo
from std_msgs.msg import Float32
import time
from .gazebo_connection import GazeboConnection
class BaseWrapperROS(gym.Env):
    def __init__(self, rate):
        """
        Init the gazebo ros
        Init nodes and topics
        Subscribers & publishers
        """
        self.wrapper_node   =   rospy.init_node('wrapper', anonymous=True)
        self.gazebo         =   GazeboConnection(True, 'WORLD')
        
        #self.reward_pub     =   rospy.Publisher('/wrapper/reward', Float32, queue_size=1)
        #self.reward_pub.publish(32.5)
        self.rate           =   rospy.Rate(rate)
        
    """
    Starts definition of private methods,
    starts with an '_' words are separate by '_' 
    and all letters are in lower case
    """
    def _set_action_msg(self, action):
        """
        Set the message actions
        """
        raise NotImplementedError()

    def _set_states(self, position=None, attitude=None):
        """
        Set the states of the quadrotor,
        if position or attitude is given
        initialize with that, otherwise
        initialize at random states descibed by a gaussian
        """
        raise NotImplementedError()
    
    def _compute_rewards(self, states_dict: Dict[str, np.ndarray], action: np.ndarray, target_pos: Optional[np.ndarray]=None) -> float:
        """
        Compute Rewards of the transition,
        given the states and the action taken
        """
        raise NotImplementedError()

    def _compute_done(self, states_dict: Dict[str, np.ndarray], target_pos: Optional[np.ndarray]=None) -> bool:
        """
        Compute done to early stop from states
        """
        raise NotImplementedError()
        

    def _get_observation_state(self) -> Dict[str, np.ndarray]:
        """
        Get the observation states flatten
        """
        raise NotImplementedError()

    
    def _flat_observation(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Flatten the observation
        """
        raise NotImplementedError()

    def _reset_sim(self):
        #self.gazebo.unpauseSim()
        #self._check_all_systems_ready() #TODO
        #self._set_states()              #TODO
        #self.gazebo.pauseSim()
        #self.gazebo.resetSim()
        #self.gazebo.unpauseSim()
        #self._check_all_systems_ready() #TODO
        #self.gazebo.pauseSim()

        self.gazebo.unpauseSim()
        self._check_all_systems_ready() #TODO
        #self._set_states()              #TODO
        self.gazebo.pauseSim()
        self.gazebo.resetSim()
        self.gazebo.unpauseSim()
        self._set_states()
        self._check_all_systems_ready() #TODO
        #self._set_states() # changed
        # line added
        self.gazebo.pauseSim()
        time.sleep(0.1)
        return True

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        raise NotImplementedError()


    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        return None
        #raise NotImplementedError()
    """ 
    Starts definition of public methods
    """

    def step(
        self, 
        action: np.ndarray, 
        targetpos: Optional[np.ndarray]=None
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute the step process given an action
        @action: a numpy array of dimension (4, )
        
        To control the time step, it is used pause & unpause
        Gazebo.
        """
        #print(action)
        self.gazebo.unpauseSim()
        self._set_action_msg(action)
        self.rate.sleep()
        
        self.gazebo.pauseSim()
        obs_dict    =   self._get_observation_state()
        done        =   self._compute_done(obs_dict, targetpos)
        info        =   {}
        reward      =   self._compute_rewards(obs_dict, action, targetpos)

        obs =   self._flat_observation(obs_dict)

        return obs, reward, done, info

    
    def _before_reset(self):
        """
        Make some stuff before reset the simulator
        """
        pass

    def reset(self):
        """
        Resets the Ros enverionment and is in charge of initialization
        """
        self._before_reset()
        self._reset_sim()
        self._init_env_variables()
        #TODO: update episode
        obs_dict    =   self._get_observation_state()
        obs         =   self._flat_observation(obs_dict)
        return obs


    def close(self):
        """
        Close the gazebo simulator
        """
        rospy.signal_shutdown('Closing WrapperROS')

    def render(self, close=False):
        """
        Render to visualize if it is possible to execute in hidden mode
        """
        pass
