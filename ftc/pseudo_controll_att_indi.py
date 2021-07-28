"""
    Position Controller based on PID
"""
import numpy as np
from math import sin, cos
from ftc.parameters import Parameters

class PseudoControllAttINDI:
    def __init__(self, parameters: Parameters):
        self.mass = parameters.mass
        self.chi = parameters.chi/57.3
        self.gravity = parameters.gravity

        # gains
        self.kpz = parameters.pos_z_p_gain
        self.kdz = parameters.pos_z_d_gain
        self.katt_p = parameters.att_p_gain
        self.katt_d = parameters.att_d_gain
        self.kpr = parameters.YRC_Kp_r

    def __call__(self, state, n_des, lambda_, nB, r_ref, Z_ref, Vz_ref):
        phi = state.att[0]
        theta = state.att[1]
        psi = state.att[2]
        
        p = state.omegaf[0]
        q = state.omegaf[1]
        r = state.omegaf[2]

        vZ = state.vel[2]
        Z = state.pos[2]

        lambda_ = lambda_.flatten()
        nB = nB.flatten()
        nx = nB[0]
        ny = nB[1]

        chi = self.chi
        if state.fail_id == 0 or state.fail_id == 2: 
            chi = np.pi - self.chi
        
        ################################

        Rib = np.array([
            [np.cos(psi) * np.cos(theta), np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi), np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)],
            [np.sin(psi) * np.cos(theta) , np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi), np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)],
            [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)]
        ])
        #h = np.matmul(np.linalg.inv(Rib), n_des)
        h = np.linalg.lstsq(Rib, n_des)[0].flatten()

        h1, h2, h3 = h[0], h[1], h[2]

        Y = np.array([[h1 - nx], [h2 - ny]])

        dY = np.array([
            [-h3 * q + h2 * r + lambda_[0]],
            [h3 * p + h1 * r + lambda_[1]]
        ])

        local_mat = np.array([
                        [cos(chi), sin(chi)], 
                        [-sin(chi), cos(chi)]
                    ])
        Y = np.matmul(local_mat, Y)
        dY = np.matmul(local_mat, dY)

        nu1 = - self.kdz * (vZ - Vz_ref) - self.kpz * (Z - Z_ref)
        nu2 =  - self.katt_d * dY[0] - self.katt_p * Y[0]
        nu3 =  - self.katt_d * dY[1] - self.katt_p * Y[1]
        nu4 =  - self.kpr * (r-r_ref)

        nu = np.array([
            [nu1], [nu2[0]], [nu3[0]], [nu4],
        ])

        return nu, dY, Y

if __name__ == '__main__':
    params = Parameters()#'../params/quad_parameters.json', '../params/control_parameters.json')
    pc = PseudoControllAttINDI(params)
    from ftc.state import State
    state = State()
    state.update({
        'position': np.array([0.0019, -0.0030, -0.0013]),
        'quaternion': None,
        'linear_vel': np.array([-0.0047, -0.0008, -0.0048]),
        'angular_vel': np.array([-0.2015, -0.2467, -0.0825]),
        'rotation_matrix': None,
        'euler': np.array([0.41416e-03, -0.17189e-03, 0.2822e-03]),
        'lin_acc': None,
        'w_speeds': None
    })

    state.update_fail_id(2)
    n_des = np.array([0.0004, 0.0011, -1.0]).reshape(-1, 1)
    _lambda = np.array([0.3821, 0.5, -0.5]).reshape(-1, 1)
    nB = np.array([0, 0, -1]).reshape(-1, 1)
    r_ref = - 0.0013
    Z_ref = 0
    Vz_ref = 0

    nu, dY, Y = pc(state, n_des, _lambda, nB, r_ref, Z_ref, Vz_ref)
    
    #expected
    #nu = np.array([0.0206, -21.5172, -1.5166, 0.4058]).reshape(-1, 1)
    # dY = np.array([0.7126, 0.0508]).reshape(-1, 1)
    # Y = np.array([0.6914e-03, -0.0327e-03]).reshape(-1, 1)
        

