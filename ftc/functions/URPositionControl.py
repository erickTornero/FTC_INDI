import numpy as np
#errorInt = [0, 0, 0]
def URpositionControl(inputs, state, par, errorInt):
    maxAngle = par.position_maxAngle

    #position control
    errorPos = [inputs.xTarget, inputs.yTarget, inputs.zTarget]  - state.posf

    velTarget = par.position_Kp_pos * errorPos
    maxVel = par.position_maxVel
    velTarget = np.clip(velTarget,-maxVel, maxVel)
    state.vel_ref[:2] = velTarget[:2]
    state.pos_ref = [inputs.xTarget, inputs.yTarget, inputs.zTarget]

    # velocity control
    errorVel = velTarget - state.velf
    errorInt = errorInt + errorVel/par.freq
    intLim = par.position_intLim
    errorInt = np.clip(errorInt,-intLim, intLim)

    # reference acceleration
    a_ref = par.position_Kp_vel * errorVel + par.position_Ki_vel * errorInt
    a_ref[2] = a_ref[2] - par.g

    maxLateral = np.abs(par.g*np.tan(maxAngle))
    latRatio = np.sqrt(a_ref[0]**2 + a_ref[1]**2)/maxLateral
    a_ref[0] = a_ref[0]/(max(latRatio,1))
    a_ref[1] = a_ref[1]/(max(latRatio,1))

    # normalise
    n_des = a_ref/np.linalg.norm(a_ref)

    return n_des