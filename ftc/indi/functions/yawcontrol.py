from math import cos, sin

def URYawControl(inputs, state, par):
    psiError = inputs.yawTarget - state.att[2]
    psi_dot_cmd = psiError * par.YRC_Kp_psi

    r_cmd = psi_dot_cmd*cos(state.att[0])*cos(state.att[1])-sin(state.att[0])*state.omegaf[1]

    return r_cmd