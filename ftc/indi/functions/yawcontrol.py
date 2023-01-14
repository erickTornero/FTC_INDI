from math import cos, sin
from ftc.utils.state import State
from ftc.utils.inputs import Inputs
from ftc.indi.parameters import Parameters

def URYawControl(inputs: Inputs, state: State, par: Parameters):
    psiError = inputs.yaw_target - state.att[2]
    psi_dot_cmd = psiError * par.YRC_Kp_psi

    r_cmd = psi_dot_cmd*cos(state.att[0])*cos(state.att[1])\
                -sin(state.att[0])*state.omegaf[1]

    return r_cmd