from math import cos, sin
from ftc.utils.state import State
from ftc.utils.inputs import Inputs
from ftc.lqr.parameters import Parameters

def yaw_controller(inputs: Inputs, state: State, par: Parameters) -> float:
    psiError = inputs.yaw_target - state.att[2]
    psi_dot_cmd = psiError * par.YRC_Kp_psi

    r_cmd = psi_dot_cmd*cos(state.att[0])*cos(state.att[1])\
                -sin(state.att[0])*state.omegaf[1]

    return r_cmd

#TODO: check attitude calculation 