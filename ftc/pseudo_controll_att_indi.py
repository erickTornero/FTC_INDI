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

        nx = nB[0]
        ny = nB[1]

        chi = self.chi
        if state.fail_id == 1 or state.fail_id == 3: 
            chi = np.pi - self.chi
        
        ################################

        Rib = np.array([
            [np.cos(psi) * np.cos(theta), np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi), np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)],
            [np.sin(psi) * np.cos(theta) , np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi), np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)],
            [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)]
        ])
        h = np.matmul(np.linalg.inv(Rib), n_des)

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
    pc = PseudoControllAttINDI(1, 1, 9.81, 0.1, 0.1, 0.1, 0.1, 0.1)
    
    response = pc(
        np.array([0, 0,0.8 ]),
        np.array([1, 0.1, 8 ]),
        np.array([0, 0, 1 ]),
        np.array([-0.5, 0.8, 1.2 ]),
        np.array([1, 1.5, 0.8 ]),
        np.array([1, 1, 1 ]),
    )
    
    print(response)
        

