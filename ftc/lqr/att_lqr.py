import control
import numpy as np
from .equilibrium_sim import SolutionQuadrotorWrapper
from ftc.lqr.parameters import Parameters
from ftc.utils.state import State
from math import cos, sin
class ReducedAttitudeController:
    def __init__(self, parameters: Parameters, alpha_ratio=0.5, double_rotor=False):
        self.fail_id = parameters.fail_id
        self.parameters = parameters
        #
        solver = SolutionQuadrotorWrapper(
            mass=parameters.mass,
            gravity=parameters.gravity,
            kt=parameters.kt,
            kf=parameters.kf,
            dumping=parameters.aerodynamics_damping,
            length_arm=parameters.arm_length,
            izzt=parameters.izzt,
            ixxt=parameters.ixxt,
            izzp=parameters.izzp,
            ixxb=parameters.ixxb
        )
        Ae, Be, solution = solver.get_extended_control_matrixes(
            parameters.fail_id + 1,
            alpha_ratio=alpha_ratio,
            up_time_motor=0.01875,#parameters.actuator_dynamics
            double_rotor=double_rotor
        )
        self.equilibrium_state = solution
        self.u1_equilibrium = self.equilibrium_state.f3 - self.equilibrium_state.f1
        self.u2_equilibrium = self.equilibrium_state.f2 - self.equilibrium_state.f4

        #Q  =   np.diag([0, 0, 2, 2, 0.0, 0.0])
        Q = np.diag([0, 0, 2, 2] + ([0,] if double_rotor else [0, 0]))
        #Q  =   np.diag([0.0, 0.0, 1, 1, 0.0, 0.0])
        R  =   np.eye(2 if not double_rotor else 1)

        K, _, _ = control.lqr(Ae, Be, Q, R)

        self.K_lqr = K
        # other params
        self.Iz =   parameters.Iz
        self.kf =   parameters.kf
        self.kt =   parameters.kt
        self.length_arm = parameters.arm_length
        self.ixxb = parameters.ixxb

        # experimental
        matrix = np.array([
            [-1, 0, 1, 0],   # f3 - f1
            [0, 1, 0, -1],   # f2 - f4
            [1, 1, 1, 1],    # f1 + f2 + f3 + f4
        ])
        matrix[:, parameters.fail_id]=0.0
        self.counter_rotor_index = (parameters.fail_id + 2) % 4
        if double_rotor:
            matrix[:, self.counter_rotor_index] = 0.0
        self.AlS = np.linalg.pinv(matrix)
        self.counter_rotor_activated = False
        self.double_rotor = double_rotor

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

        h_des = np.linalg.solve(R_IB, n_des) #TODO: debug more here! hdes - h
        #import pdb; pdb.set_trace()
        #eta = np.matmul(self.R_xy_uv, h[:2].reshape(-1, 1))
        eta = h_des[:2]#.reshape(-1, 1)

        w_speeds = state.w_speeds # s: 0.34?
        # torques xy in body axis

        u1 = self.kf * (w_speeds[2]**2 - w_speeds[0]**2)
        u2 = self.kf * (w_speeds[1]**2 - w_speeds[3]**2)

        state_eq_pt = [
            u - self.equilibrium_state.p,
            v - self.equilibrium_state.q,
            eta[0] - self.equilibrium_state.nx,
            eta[1] - self.equilibrium_state.ny,
        ]
        if self.double_rotor:
            state_eq_pt = state_eq_pt + [
                (u1 - self.u1_equilibrium) if self.fail_id in [1, 3] else \
                (u2 - self.u2_equilibrium)
            ]
        else:
            state_eq_pt = state_eq_pt + [
                u1 - self.u1_equilibrium,
                u2 - self.u2_equilibrium,
            ]
        state_eq_pt = np.array(state_eq_pt).reshape(-1, 1)
        #Muv = - np.matmul(self.K_lqr, np.array([u - self.u0, v - self.v0, eta[0, 0], eta[1, 0], Muv0[0, 0], Muv0[1, 0]]).reshape(-1, 1))
        Muv = -np.matmul(self.K_lqr, state_eq_pt) # control givven a point
        T_ref = f_ref

        ## Transform to torques
        # Muv(0): f2 - f4
        # Muv(1): f3 - f1 
        # to real forces
        # Muv(0): f2 - f4 + (f2_bar - f4_bar)
        # Muv(1): f2 - f4 + (f2_bar - f4_bar)


        if self.double_rotor:
            if self.fail_id in [1, 3]: #rotor 2, 4
                alphas = np.array([
                    Muv[0, 0] + self.u1_equilibrium,
                    0.0,
                    T_ref
                ])
            else:
                alphas = np.array([
                    0.0,
                    Muv[0, 0] + self.u2_equilibrium,
                    T_ref
                ])
        else:
            # new system
            alphas = np.array([
                Muv[0, 0] + self.u1_equilibrium, # f3 - f1
                Muv[1, 0] + self.u2_equilibrium, # f2 - f4
                T_ref,                           # f1 + f2 + f3 + f4
            ])
        alphas = alphas.reshape(-1, 1)

        forces = np.matmul(
            self.AlS,
            alphas
        ).flatten()
        ##TODO: experimental all forces must be greater than 0
        forces = np.maximum(forces, 0.0)

        return forces