from typing import Dict, Optional
import numpy as np
from ftc.indi.controller import DiscreteTimeDerivative
from ftc.lqr.reduced_lqr import ReducedAttitudeController

from ftc.utils.filters import LowpassFilter
from ftc.utils.transforms import pos_invert_yz
from ftc.base_controller import BaseController
from ftc.lqr.parameters import Parameters
from wrapper.state_space import StateSpaceRobots
from ftc.lqr.calculate_lqr import Mixer, get_lqr_matrix, FlotCalculator
from ftc.utils.state import State
from ftc.utils.inputs import Inputs
from ftc.indi.functions.poscontrol import URPositionControl
from ftc.indi.functions.yawcontrol import URYawControl
class LQRController(BaseController):
    def __init__(self, parameters: Parameters, T_sampling:float=None, state_space: Optional[StateSpaceRobots]=None):
        self.T_sampling = T_sampling
        self.parameters = parameters
        self.state_space    =   state_space
        self.parameters.k_lqrff = get_lqr_matrix(parameters,-1, False)
        # single rotor failure
        self.parameters.k_lqr0 = get_lqr_matrix(parameters, 0, False)
        self.parameters.k_lqr1 = get_lqr_matrix(parameters, 1, False)
        self.parameters.k_lqr2 = get_lqr_matrix(parameters, 2, False)
        self.parameters.k_lqr3 = get_lqr_matrix(parameters, 3, False)
        # double rotor failure
        self.parameters.k_lqr02 = get_lqr_matrix(parameters, 0, True)
        self.parameters.k_lqr13 = get_lqr_matrix(parameters, 1, True)

        self.z_target_filter   = LowpassFilter(1, parameters.t_filter, T_sampling)
        self.z_target_derivator =   DiscreteTimeDerivative(T_sampling)
        self.flot_calculator    =   FlotCalculator(parameters)
        self.reduced_att_controller =   ReducedAttitudeController(parameters)
        self.mixer                  =   Mixer(parameters)

        self.errorInt = np.zeros(3, dtype=np.float32)

    def get_action(
        self, 
        obs: np.ndarray, 
        targetpos: np.ndarray, 
        obs_dict: Optional[Dict[str, np.ndarray]]=None
    ) -> np.ndarray:
        targetpos = pos_invert_yz(targetpos)
        if obs_dict is not None:
            self._state.update(obs_dict)
        else:
            self._state.update(self.state_space.get_obs_dict(obs))

        self._inputs.update_position_target(targetpos)
        control_signal = self.__call__(self._state, self._inputs)
        # swap control signal
        tmp = control_signal[3]
        control_signal[3] = control_signal[1]
        control_signal[1] = tmp

        #control_signal[0] = control_signal[0] * -1
        return control_signal

    def __call__(self, state: State, inputs: Inputs):
        """Implement here a forward pass of the controller"""
        n_des, r_cmd = self.outer_controller(state, inputs)
        f_ref = self._get_flot_lqr(state, inputs)
        U_lqr   =   self.reduced_att_controller(state, n_des, f_ref, r_cmd)
        return self.mixer(U_lqr)

    def outer_controller(self, state: State, inputs: Inputs):
        n_des, self.errorInt = URPositionControl(inputs, state, self.parameters, self.errorInt)
        r_cmd = URYawControl(inputs, state, self.parameters)
        return n_des, r_cmd


    def _get_flot_lqr(self, state: State, inputs: Inputs) -> float:
        z_target = inputs.zTarget
        zf_target = self.z_target_filter(z_target)
        vzf_target = self.z_target_derivator(zf_target)
        return self.flot_calculator(state, z_target, vzf_target)

    def init_controller(self, state0: State, inputs: Inputs, ct: float, damaged_motor=2):
        """inputs = inputs.
        position_target = pos_invert_yz(position_target)
        # initialize state and inputs
        _state = State(invert_axis=True)
        _state.update_fail_id(damaged_motor)
        _input = Inputs()
        _input.update_position_target(position_target)
        _input.update_yawTarget(0)
        inputs = _input
        states = _state

        self.z_target_filter.start(inputs.zTarget, ct)
        #self.low_pass_ndes.start(np.array([0, 0, -1]).reshape(-1,1), t)

        # derivators
        
        self.z_target_derivator.start(states.position[2])

        #ssres = self.subsystem(states, n_des)
        
        #h0, posdd, U0, U1 = ssres['h0'], ssres['posdd'], ssres['U0'], ssres['U1']
        self._state = states
        self._inputs = inputs

        self.ndes_list  =   []
        self.yaw_speed_cmd    =   []
        """