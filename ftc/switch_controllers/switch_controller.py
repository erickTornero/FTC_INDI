from ftc.ffree import HoverController
from ftc.indi import INDIController
from ftc.lqr.controller import LQRController
from wrapper import QuadrotorEnvRos
from ftc.lqr.parameters import Parameters as ParametersLQR
from ftc.indi.parameters import Parameters
from enum import Enum
import numpy as np

class MODE_CONTROLLER(Enum):
    FAULT_FREE = 1
    FTC_CASE = 2

class FTC_CONTROLLER_TYPE(Enum):
    INDI=1
    LQR=2

class SwitchController:
    def __init__(
            self,
            env: QuadrotorEnvRos,
            initial_mode=MODE_CONTROLLER.FAULT_FREE, 
            ftc_controller_type: FTC_CONTROLLER_TYPE=FTC_CONTROLLER_TYPE.INDI
    ) -> None:
        self.ffree_controller = HoverController(env.state_space)
        if ftc_controller_type == FTC_CONTROLLER_TYPE.INDI:
            parameters = Parameters(
                'params/quad_parameters.json',
                'params/control_parameters_indi.json'
            )
            self.ftc_controller = INDIController(
                parameters=parameters,
                T_sampling=1/parameters.freq,
                state_space=env.state_space
            )
        elif ftc_controller_type == FTC_CONTROLLER_TYPE.LQR:
            parameters = ParametersLQR(
                'params/quad_parameters.json',
                'params/control_params_lqr.json'
            )
            self.ftc_controller = LQRController(
                parameters=parameters,
                T_sampling=1/parameters.freq,
                state_space=env.state_space
            )
        self.mode=initial_mode
        self.env = env # for track reference
        self.controller = self.ffree_controller

    def get_action(self, obs: np.ndarray, target_pos: np.ndarray)->np.ndarray:
        return self.controller.get_action(obs, target_pos, self.env.last_observation)

    def switch_faulted(self, position_target: np.ndarray, damaged_rotor_index: int, c_time: float):
        self.ftc_controller.init_controller(
            self.env.last_observation, position_target, damaged_rotor_index, c_time
        )
        self.controller = self.ftc_controller
        self.mode = MODE_CONTROLLER.FTC_CASE
        print('Switching controller to faulted case')

    def switch_fault_free(self):
        self.controller = self.ffree_controller
        self.mode = MODE_CONTROLLER.FAULT_FREE