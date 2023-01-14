from typing import Dict, Optional
import numpy as np
from ftc.indi.controller import DiscreteTimeDerivative

from ftc.utils.filters import LowpassFilter
from ftc.base_controller import BaseController
from ftc.lqr.parameters import Parameters
from wrapper.state_space import StateSpaceRobots
from ftc.lqr.calculate_lqr import Mixer, FlotCalculator
from ftc.utils.state import State
from ftc.utils.inputs import Inputs
from ftc.lqr.yawcontrol import yaw_controller
from ftc.lqr.position_controller import PositionController
from ftc.lqr.att_lqr import ReducedAttitudeController
class LQRController(BaseController):
    def __init__(self, parameters: Parameters, T_sampling:float=None, state_space: Optional[StateSpaceRobots]=None):
        self.T_sampling = T_sampling # fast inner Ts controller
        self.parameters = parameters
        self.state_space    =   state_space

        self.z_target_filter   = LowpassFilter(1, parameters.t_filter, T_sampling)
        self.z_target_derivator =   DiscreteTimeDerivative(T_sampling)
        self.flot_calculator    =   FlotCalculator(parameters)

        self.double_rotor = parameters.double_rotor
        self.red_att_double = ReducedAttitudeController(
            parameters,
            alpha_ratio=0.0,
            double_rotor=True
        )
        if not self.double_rotor:
            self.red_att_single = ReducedAttitudeController(
                parameters,
                alpha_ratio=parameters.alpha_ratio,
                double_rotor=self.double_rotor
            )
            self.counter_rotor_activated = False

        self.mixer = Mixer(parameters)

        self.position_controller = PositionController(
            T_sampling,
            10,
            parameters.gravity,
            parameters.position_Kp_pos,
            parameters.position_Kp_vel,
            parameters.position_Ki_vel,
            parameters.position_maxAngle,
            parameters.position_intLim,
            parameters.position_maxVel,
        )
        self.switch_yaw_rate_threshold = 10.0

    def get_action(
        self, 
        obs: np.ndarray, 
        targetpos: np.ndarray, 
        obs_dict: Optional[Dict[str, np.ndarray]]=None
    ) -> np.ndarray:
        #targetpos = pos_invert_yz(targetpos)
        if obs_dict is not None:
            self._state.update(obs_dict)
        else:
            self._state.update(self.state_space.get_obs_dict(obs))

        self._inputs.update_position_target(targetpos)
        control_signal = self.__call__(self._state, self._inputs)
        return control_signal

    def __call__(self, state: State, inputs: Inputs):
        """Implement here a forward pass of the controller"""
        n_des, r_cmd = self.outer_controller(state, inputs)
        f_ref = self._get_flot_lqr(state, inputs)
        yaw_rate_abs: float = np.abs(state.yaw_rate)

        if self.double_rotor or yaw_rate_abs <= self.switch_yaw_rate_threshold:
            U_lqr = self.red_att_double(state, n_des, f_ref, r_cmd)
        else:
            U_lqr = self.red_att_single(state, n_des, f_ref, r_cmd)
            if not self.counter_rotor_activated:
                print(' ... Using single rotor lqr controller ...')
                self.counter_rotor_activated = True
        return self.mixer(U_lqr)

    def outer_controller(self, state: State, inputs: Inputs):
        n_des = self.position_controller(state, inputs)
        r_cmd = yaw_controller(inputs, state, self.parameters)
        return n_des, r_cmd


    def _get_flot_lqr(self, state: State, inputs: Inputs) -> float:
        z_target = inputs.z_target
        zf_target = self.z_target_filter(z_target)
        vzf_target = self.z_target_derivator(zf_target) # not working well #Check initial, recently changed
        return self.flot_calculator(state, z_target, vzf_target)

    def init_controller(self, state0: State, inputs: Inputs, ct: float, damaged_motor=2):
        #inputs = inputs.
        position_target = np.array([inputs.x_target, inputs.y_target, inputs.z_target])
        #position_target = pos_invert_yz(position_target)
        # initialize state and inputs
        _state = State(invert_axis=False)  # Duplicated state, tends to error
        _state.update_fail_id(damaged_motor)
        _input = Inputs()
        _input.update_position_target(position_target)
        _input.update_yaw_target(0)
        inputs = _input
        states = _state

        self.z_target_filter.start(inputs.z_target, ct)
        #self.low_pass_ndes.start(np.array([0, 0, -1]).reshape(-1,1), t)

        # derivators
        
        #self.z_target_derivator.start(state0.position[2]) # should be input
        self.z_target_derivator.start(inputs.z_target) # should be input

        #ssres = self.subsystem(states, n_des)
        
        #h0, posdd, U0, U1 = ssres['h0'], ssres['posdd'], ssres['U0'], ssres['U1']
        self._state = states
        self._inputs = inputs

        self.ndes_list = []
        self.yaw_speed_cmd = []
        