import numpy as np
from ftc.lqr.parameters import Parameters
from ftc.utils.state import State
from math import cos, sin
import control
from .equilibrium_sim import SolutionQuadrotorWrapper
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
        #Mz_ref = self.yrc_kp_r * (r - r_cmd) * self.Iz
        #Mz_ref = self.yrc_kp_r * r # improved computation of the momment z #improve
        #must be computed with damping
        Mz_ref = 2.75e-3 * r

        U = np.zeros(4)
        U[0] = Muv[0, 0]
        U[1] = Muv[1, 0]
        U[2] = T_ref            # total sum of force
        U[3] = Mz_ref           # total moment
        return U

class ReducedAttitudeControllerImproved:
    def __init__(self, parameters: Parameters, alpha_ratio=0.5):
        self.fail_id = parameters.fail_id
        self.parameters = parameters
        #
        self.u0 = 0
        self.v0 = 0
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
            parameters.fail_id + 1,
            alpha_ratio=alpha_ratio,
            up_time_motor=parameters.actuator_dynamics
        )
        self.equilibrium_state = solution
        self.u1_equilibrium = self.equilibrium_state.f3 - self.equilibrium_state.f1
        self.u2_equilibrium = self.equilibrium_state.f2 - self.equilibrium_state.f4

        Q  =   np.diag([0, 0, 2, 2, 0.0, 0.0])
        R  =   np.eye(2)

        K, _, _ = control.lqr(Ae, Be, Q, R)

        self.K_lqr = K
        # other params
        self.Iz =   parameters.Iz
        self.kf =   parameters.kf
        self.kt =   parameters.kt
        self.length_arm = parameters.arm_length
        self.ixxb = parameters.ixxb

        # experimental
        self.AlS = np.linalg.inv(np.array([
            [-1, 0, 0],
            [0, 1, -1],
            [1, 1, 1]
        ]))

    def __call__(self, state: State, n_des: np.ndarray, f_ref: float, r_cmd: float) -> np.ndarray:
        # parameters from state
        # attitude, omegaf, w_speeds
        phi = state.att[0]
        theta = state.att[1]
        psi = state.att[2]

        u = state.omegaf[0]# tmp[0, 0] # u1 # assumption this is pqr is it in body frame?
        v = state.omegaf[1]# tmp[1, 0] # u2
        r = state.omegaf[2]

        R_IB = np.array([
            [cos(psi) * cos(theta), cos(psi) * sin(theta) * sin(phi) - sin(psi) * cos(phi), cos(psi) * sin(theta) * cos(phi) + sin(psi) * sin(phi)],
            [sin(psi) * cos(theta), sin(psi) * sin(theta) * sin(phi) + cos(psi) * cos(phi), sin(psi) * sin(theta) * cos(phi) - cos(psi) * sin(phi)],
            [-sin(theta), cos(theta) * sin(phi), cos(theta) * cos(phi)]
        ])
        #u, v, _ = np.matmul(R_IB, np.array(state.omegaf).reshape(-1, 1)).flatten().tolist()

        h_des = np.linalg.solve(R_IB, n_des)
        #import pdb; pdb.set_trace()
        #eta = np.matmul(self.R_xy_uv, h[:2].reshape(-1, 1))
        eta = h_des[:2]#.reshape(-1, 1)

        w_speeds = state.w_speeds # s: 0.34?
        # torques xy in body axis
        #Muv0 = self.kf * self.length_arm * np.array([w_speeds[1]**2 - w_speeds[3]**2, w_speeds[2]**2 - w_speeds[0]**2])
        Muv0 = self.kf * np.array([w_speeds[1]**2 - w_speeds[3]**2, w_speeds[2]**2 - 405**2])
        Muv0 = -self.kf * np.array([w_speeds[2]**2 - w_speeds[0]**2, w_speeds[1]**2 - 405**2])

        Muv0 = self.kf * np.array([w_speeds[2] - w_speeds[0]**2, w_speeds[1]**2 - w_speeds[3]**2])
        ###Muv0 -= np.array([self.u1_equilibrium, self.u2_equilibrium])
        ##Muv0 = self.kf * np.array([w_speeds[1]**2 - w_speeds[3]**2, w_speeds[2] - w_speeds[0]**2]) #...from equation 12 and 13
        ##Muv0 -= np.array([self.u2_equilibrium, self.u1_equilibrium, ])
        #Muv0 = self.kf * np.array([w_speeds[2]**2 - w_speeds[0]**2, w_speeds[1]**2 - 405**2])
        #Muv0 = self.k0 * self.s * np.array([[w_speeds[3]**2 - w_speeds[1]**2], [w_speeds[0]**2 - w_speeds[2]**2]])
        state_eq_pt = np.array([
            u - self.equilibrium_state.p,
            v - self.equilibrium_state.q,
            eta[0] - self.equilibrium_state.nx,
            eta[1] - self.equilibrium_state.ny,
            Muv0[0] - self.u1_equilibrium,
            Muv0[1] - self.u2_equilibrium,
        ]).reshape(-1, 1)
        #Muv = - np.matmul(self.K_lqr, np.array([u - self.u0, v - self.v0, eta[0, 0], eta[1, 0], Muv0[0, 0], Muv0[1, 0]]).reshape(-1, 1))
        Muv = -np.matmul(self.K_lqr, state_eq_pt) # control givven a point
        T_ref = f_ref
        #Mz_ref = self.yrc_kp_r * (r - r_cmd) * self.Iz
        #Mz_ref = self.yrc_kp_r * r # improved computation of the momment z #improve
        #must be computed with damping
        Mz_ref = 2.75e-3 * r

        ## Transform to torques
        # Muv(0): f2 - f4
        # Muv(1): f3 - f1 
        # to real forces
        # Muv(0): f2 - f4 + (f2_bar - f4_bar)
        # Muv(1): f2 - f4 + (f2_bar - f4_bar)
        # transform to torques
        # Tx = (f2 - f4) * l_arm
        # Ty = (f3 - f1) * l_arm
        #####
        # u1: f3 - f1 + (f3_bar - f1_bar)
        # u2: f2 - f4 + (f2_bar - f4_bar)
        # Tx = (u2 + u2_bar)*length_arm
        # Ty = (u1 + u1_bar)*length_arm
        Tx = (Muv[1, 0] + self.u2_equilibrium) * self.length_arm
        Ty = (Muv[0, 0] + self.u1_equilibrium) * self.length_arm
        #print(Mz_ref)
        U = np.zeros(4)
        U[0] = Tx #u1 (f2  - f4) 
        U[1] = Ty #u2 (f3 - f1)
        U[2] = T_ref            # total sum of force
        U[3] = Mz_ref           # total moment

        # new system
        
        f1, f2, f4 = np.matmul(
            self.AlS,
            np.array([
                Muv[0, 0] + self.u1_equilibrium,
                Muv[1, 0] + self.u2_equilibrium,
                T_ref,
                #self.equilibrium_state.f1 + self.equilibrium_state.f2 + self.equilibrium_state.f3 + self.equilibrium_state.f4
            ]).reshape(-1, 1)
        ).flatten()
        return np.array([f1, f2, 0, f4])
        
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