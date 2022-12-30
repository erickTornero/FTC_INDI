import numpy as np
from ftc.lqr.parameters import Parameters
from ftc.utils.state import State
from math import cos, sin

class ReducedAttitudeController:
    def __init__(self, parameters: Parameters):
        K = parameters.k_lqrff
        self.fail_id = parameters.fail_id
        self.parameters = parameters

        # 
        self.u0 = 0
        self.v0 = 0
        self.axis_tilt = parameters.axis_tilt
        self.DRF_enable = parameters.DRF_enable

        if self.fail_id >= 0:
            if not self.DRF_enable:
                if self.fail_id == 0:
                    K = self.parameters.k_lqr0
                    self.u0 = 0
                    self.v0 = -self.axis_tilt
                elif self.fail_id == 1:
                    K = self.parameters.k_lqr1
                    self.u0 = - self.axis_tilt
                    self.v0 = 0
                elif self.fail_id == 2:
                    K = self.parameters.k_lqr2
                    self.u0 = 0
                    self.v0 = self.parameters.axis_tilt
                elif self.fail_id == 3:
                    K = self.parameters.k_lqr3
                    self.u0 = self.axis_tilt
                    self.v0 = 0
            else:
                if self.fail_id == 0 or self.fail_id == 2:
                    K = self.parameters.k_lqr02
                else:
                    K = self.parameters.k_lqr13
        
        self.K_lqr = K
        # other params
        self.Iz =   parameters.Iz
        self.k0 =   parameters.k0
        self.s  =   parameters.s
        self.yrc_kp_r = parameters.YRC_Kp_r
        
        R_xy_uv    =   np.linalg.inv(np.array([[parameters.l, parameters.l], [-parameters.b, parameters.b]]))
        self.R_xy_uv    =   R_xy_uv/np.linalg.norm(R_xy_uv)
    
    def __call__(self, state: State, n_des: np.ndarray, f_ref: float, r_cmd: float) -> np.ndarray:
        # parameters from state
        # attitude, omegaf, w_speeds
        phi = state.att[0]
        theta = state.att[1]
        psi = state.att[2]
        tmp = np.matmul(self.R_xy_uv, np.array([state.omegaf[0], state.omegaf[1]]).reshape(-1, 1))

        u = tmp[0, 0] # u1
        v = tmp[1, 0] # u2
        r = state.omegaf[2]

        R_IB = np.array([
            [cos(psi) * cos(theta), cos(psi) * sin(theta) * sin(phi) - sin(psi) * cos(phi), cos(psi) * sin(theta) * cos(phi) + sin(psi) * sin(phi)],
            [sin(psi) * cos(theta), sin(psi) * sin(theta) * sin(phi) + cos(psi) * cos(phi), sin(psi) * sin(theta) * cos(phi) - cos(psi) * sin(phi)],
            [-sin(theta), cos(theta) * sin(phi), cos(theta) * cos(phi)]
        ])

        h = np.linalg.solve(R_IB, n_des)
        eta = np.matmul(self.R_xy_uv, h[:2].reshape(-1, 1))

        w_speeds = state.w_speeds # s: 0.34?
        Muv0 = self.k0 * self.s * np.array([[w_speeds[3]**2 - w_speeds[1]**2], [w_speeds[0]**2 - w_speeds[2]**2]])

        Muv = - np.matmul(self.K_lqr, np.array([u - self.u0, v - self.v0, eta[0, 0], eta[1, 0], Muv0[0, 0], Muv0[1, 0]]).reshape(-1, 1))

        T_ref = f_ref
        Mz_ref = -self.yrc_kp_r * (r - r_cmd) * self.Iz

        U = np.zeros(4)
        U[0] = Muv[0, 0]
        U[1] = Muv[1, 0]
        U[2] = T_ref            # total sum of force
        U[3] = Mz_ref           # total moment
        return U

if __name__ == "__main__":
    from ftc.lqr.calculate_lqr import get_lqr_matrix
    parameters = Parameters('../../params/quad_parameters.json', '../../params/control_params_lqr.json')
    parameters.k_lqrff = get_lqr_matrix(parameters,-1, False);
    parameters.k_lqr2 = get_lqr_matrix(parameters, 2, False);

    rat = ReducedAttitudeController(parameters)

    class DummyState:
        def __init__(self):
            self.att = [0.1, 0.3, -0.02]
            self.omegaf = [2.1, 0.8, -3]
            self.w_speeds = [300, 200, 400 ,500]
    
    ndes = np.array([-0.1, 0.0, -0.98])
    U = rat(DummyState(), ndes, 3, 30)
    print(U)