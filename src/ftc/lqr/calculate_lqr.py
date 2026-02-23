from operator import mod
from typing import List, Union
import numpy as np
import control
from .parameters import Parameters
from ..utils.state import State
from .equilibrium_sim import SolutionQuadrotorWrapper
def get_lqr_matrix(parameters: Parameters, fail_idx: Union[int, List[int]], DRF_enable: bool):
    actuator_dynamics = parameters.actuator_dynamics
    arm_length          =   parameters.arm_length
    if DRF_enable and fail_idx>=0:
        if fail_idx == 0 or fail_idx == 2:
            fail_id = [0, 2]
        elif fail_idx == 1 or fail_idx == 3:
                fail_id = [1, 3]
    else:
        fail_id = fail_idx;

    # no failure
    if fail_id == -1:
        r_bar = 0
        w_bar = np.ones((4, )) * 460
    elif type(fail_id) == int:
        r_bar = (2 * (mod(fail_id, 2)) -1) * 20
        w_bar   =   838 * np.ones((4, ))
        w_bar[fail_id] = 0
    elif type(fail_id) == list and len(fail_id) == 2:
        r_bar   =   (2 * (np.mod(np.sum(fail_id)/2, 2)) - 1) * 20
        w_bar   =   838 * np.ones((4, ))
        w_bar[fail_id] = 0
    else:
        pass
    """
    Iz = parameters.Iz
    Iu = parameters.Iu
    Iv = parameters.Iv

    Au  =   (Iv - Iz)/Iu
    Av  =   (Iz - Iu)/Iv
    Ip  =   parameters.Ip
    k   =   parameters.act_dyn
    a   =   Au * r_bar - Ip * (w_bar[0] - w_bar[1] + w_bar[2] - w_bar[3])/Iu
    b   =   Av * r_bar + Ip * (w_bar[0] - w_bar[1] + w_bar[2] - w_bar[3])/Iv
    """
    ixxt    =   parameters.ixxt
    izzt    =   parameters.izzt
    ixxb    =   parameters.ixxb
    izzp    =   parameters.izzp

    a = ((ixxt - izzt)/ixxb) * r_bar - (izzp/ixxb) * (w_bar[0] - w_bar[1] + w_bar[2] - w_bar[3])
    A   =   np.array([
        [0, a, 0, 0],
        [-a, 0, 0, 0],
        [0, 1, 0, r_bar],
        [-1, 0, -r_bar, 0]
    ])
    B   =   (arm_length/ixxb) * np.array([
        [1, 0],
        [0, 1],
        [0, 0],
        [0, 0]
    ])

    """
    B   =   np.array([
        [1/Iu, 0],
        [0, 1/Iv],
        [0, 0],
        [0, 0]
    ])
    """

    if hasattr(fail_id, '__len__'):
        if (np.sum(fail_id) < 3): # failure 0, 2
            B[:, 1] = 0
        elif type(fail_id) == int:
            if fail_id >= 0: B[:, 0] = 0

    
    #S = k * np.eye(2) # 2x2
    S = (1/actuator_dynamics) * np.eye(2)

    tShape = (S.shape[0], A.shape[1]) # (2 x 4)
    AA = np.vstack([
        np.hstack([A, B]),                  # [(4,4),(4,2)] => (4, 6)
        np.hstack([np.zeros(tShape), -S])  # [(2,4), (2,2)] => (2, 6)
    ])      # (6, 6)

    BB = np.vstack([np.zeros((A.shape[0], S.shape[1])), S]) # vstack [(4, 2), (2, 2)] => (6, 2)

    # AA => (6, 6) BB => (6, 2)

    Q  =   np.diag([0, 0, 2, 2, 0.0, 0.0])
    R  =   np.eye(2)

    K, _, _ = control.lqr(AA, BB, Q, R)
    return K

def get_lqr_matrix_improved(parameters: Parameters, fail_idx: int, alpha_ratio: float):
    solver = SolutionQuadrotorWrapper(
        mass=parameters.mass,
        gravity=parameters.gravity,
        kt=parameters.kt,
        kf=parameters.kf,
        dumping=2.75e-3,
        length_arm=parameters.arm_length,
        izzt=parameters.izzt,
        ixxt=parameters.ixxt,
        izzp=parameters.izzp,
        ixxb=parameters.ixxb
    )
    Ae, Be, solution = solver.get_extended_control_matrixes(
        fail_idx,
        alpha_ratio=alpha_ratio,
        up_time_motor=parameters.actuator_dynamics
    )
    Q  =   np.diag([0, 0, 2, 2, 0.0, 0.0])
    R  =   np.eye(2)

    K, _, _ = control.lqr(Ae, Be, Q, R)
    return K
    

def calculate_throtle(state, z_ref, vz_ref, parameters):
    az_des = parameters.kpz_pos * (z_ref - state.pos[2]) + \
        parameters.kdz_pos * (vz_ref - state.vel[2]) - parameters.g
    
    f_tot = - az_des * parameters.mass / np.cos(state.att[0]) / np.cos(state.att[1])
    return f_tot

class FlotCalculator:
    def __init__(self, parameters: Parameters) -> None:
        self.kpz_pos    =   parameters.kpz_pos
        self.kdz_pos    =   parameters.kdz_pos
        self.g          =   parameters.gravity
        self.mass       =   parameters.mass

    def __call__(self, state: State, z_ref: float, vz_ref: float) -> float:
        az_des = self.kpz_pos * (z_ref - state.pos[2]) + \
        self.kdz_pos * (vz_ref - state.vel[2]) + self.g #TODO: possible must add g
        f_tot = az_des * self.mass / np.cos(state.att[0]) / np.cos(state.att[1])
        return f_tot

class Mixer:
    def __init__(self, parameters: Parameters) -> None:
        DRF_enable = parameters.DRF_enable
        fail_id = parameters.fail_id
        if DRF_enable and fail_id >= 0:
            if fail_id == 0 or fail_id == 2:
                fail_id = [0, 2]
            else:
                fail_id = [1, 3]
        """
        k0     =   parameters.k0
        t0     =   parameters.t0
        s      =   parameters.s/2.0

        G0  =   np.array([
            [0, 1, 0, -1], # moment X
            [-1, 0, 1, 0], # moment Y
            [1, 1, 1, 1],  # Sum Forces ...ok
            [1, -1, 1, -1] # moment Z ...ok
        ])

        G      =   np.matmul(np.diag([s * k0, s * k0, k0, t0]), G0)
        """
        kf      =   parameters.kf
        kt      =   parameters.kt
        l_arm   =   parameters.arm_length
        G       =   np.array([
            [0,           kf * l_arm, 0,          -kf*l_arm ],
            [-kf * l_arm, 0,          kf * l_arm,  0        ],
            [kf,          kf,         kf,          kf       ],
            [kt * kf,    -kt * kf,    kt * kf,    -kt * kf  ],
        ])

        if type(fail_id) == int:
            if fail_id >= 0: G[:, fail_id] = 0#np.zeros((4, 1))
        elif hasattr(fail_id, '__len__'):
            G[:, fail_id] = 0#np.zeros_like(4, len(fail_id))
        
        # matmul(G, w^2) = U
        self.G = G
        self.pinvG  =   np.linalg.pinv(G)
        self.w_min  =   parameters.w_min
        self.w_max  =   parameters.w_max
        self.kf = kf
    
    def __call__(self, U: np.ndarray) -> np.ndarray:
        w = np.sqrt(np.abs(U)/self.kf)
        w = np.clip(w, self.w_min, self.w_max)
        #print(w[0])
        #w[0] = w[0]*0.0 #TODO this an error, must work without it
        return w
        w = np.zeros(4)
        w2 = np.matmul(self.pinvG, U.reshape(-1, 1))

        w = np.sqrt(np.abs(w2))# * np.sign(w2) #TODO: Fix
        w[0] = w[0] * 0.0 #TODO: why? idx=0 must be the index
        w = np.clip(w, self.w_min, self.w_max)
        return w.flatten()

if __name__ == '__main__':
    from ftc.lqr.parameters import Parameters
    parameters = Parameters('../../params/quad_parameters.json', '../../params/control_params_lqr.json')
    # testing LQR
    print('='*20+'\n=== Testing LQR ====\n')
    K, _, Eigen = get_lqr_matrix(parameters, 0, 0)
    print(K)
    print('='*20+'\n=== Testing Mixer ====\n')

    mixer = Mixer(parameters)
    w_rotors = mixer(np.array([-30, 200, 300, 4]))
    print(w_rotors)