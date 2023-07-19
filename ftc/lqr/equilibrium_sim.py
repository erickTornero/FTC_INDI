import numpy as np
from typing import Tuple, List
from scipy.optimize import fsolve

class SolutionXBar:
    def __init__(
        self,
        nx: float,
        ny: float,
        nz: float,
        f1: float,
        f2: float,
        f3: float,
        f4: float,
        sF: float,
        E: float,
        p: float,
        q: float,
        r: float,
        w1: float,
        w2: float,
        w3: float,
        w4: float
    )  -> None:
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self.sF = sF
        self.E = E
        self.p = p
        self.q = q
        self.r = r
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

class SolutionQuadrotorWrapper:
    def __init__(
        self, 
        *,
        mass: float,
        gravity: float,
        kt: float,
        kf: float,
        dumping: float,
        length_arm: float,
        izzt: float,
        ixxt: float,
        izzp: float,
        ixxb: float,
        verbose=False,
    ) -> None:
        self.mass = mass
        self.gravity = gravity
        self.kt = kt
        self.kf = kf
        self.dumping = dumping
        self.length_arm = length_arm
        self.izzt = izzt
        self.ixxt = ixxt
        self.izzp = izzp
        self.ixxb = ixxb
        self.verbose = verbose

    def __call__(self, failed_rotor: int, alpha_ratio: float) -> Tuple[List[float], List[float]]:
        if self.verbose:
            print(f"** Solving with failed-rotor = {failed_rotor}, alpha-ratio = p = {alpha_ratio}**")
        root, cl = self.solve_generic(failed_rotor, alpha_ratio)
        if self.verbose:
            print('=====Non-linear system Error at Equilibrium Point==========')
            print("nx={:.4f}\nny={}\nnz={}\nf1={}\nf2={}\nf3={}\nf4={}\nsF={}\nE={}\np={}\nq={}\nr={}\nw1={}\nw2={}\nw3={}\nw4={}".format(*cl))
            print(f"=>Sum errors: {sum(np.abs(cl))}")
            print('===================================================\n')
            print("solution")
            print("nx={:.4f}\nny={}\nnz={}\nf1={}\nf2={}\nf3={}\nf4={}\nsF={}\nE={}\np={}\nq={}\nr={}\nw1={}\nw2={}\nw3={}\nw4={}".format(*root))
        return root, cl


    def get_control_matrixes(self, failed_rotor: int, alpha_ratio: float, double_rotor=False) -> Tuple[np.ndarray, np.ndarray, SolutionXBar]:
        if double_rotor: alpha_ratio = 0.0
        root, _ = self.__call__(failed_rotor, alpha_ratio)
        solution = SolutionXBar(*root)
        # from equation 28
        r_bar = solution.r
        nz_bar = solution.nz
        a_bar = ((self.ixxt - self.izzt)/self.ixxb) * r_bar - (self.izzp/self.ixxb) * (solution.w1 + solution.w2 + solution.w3 + solution.w4)
        
        A = np.array([
            [0, a_bar, 0, 0],
            [-a_bar, 0, 0, 0],
            [0, -nz_bar, 0, r_bar],
            [nz_bar, 0, -r_bar, 0]
        ])
        B = (self.length_arm/self.ixxb) * np.array([
            [0, 1],
            [1, 0],
            [0, 0],
            [0, 0]
        ])
        if double_rotor:
            index = failed_rotor % 2
            B = B[:, index].reshape(-1, 1)
        return A, B, solution

    def get_extended_control_matrixes(self, failed_rotor: int, alpha_ratio: float, up_time_motor: float, double_rotor=False):
        A, B, solution = self.get_control_matrixes(failed_rotor, alpha_ratio, double_rotor)
        S = np.eye(B.shape[1]) * 1/up_time_motor
        Ae_upper = np.concatenate([A, B], axis=1)
        Ae_lower = np.concatenate([np.zeros((S.shape[0], A.shape[1])), -S], axis=1)
        Ae = np.concatenate([Ae_upper, Ae_lower], axis=0)
        Be = np.concatenate([np.zeros((A.shape[0], B.shape[1])), S], axis=0)
        return Ae, Be, solution


    def get_initial_conditions(
        self,
        fail_rotor: int
    ):
        """get initial conditions"""
        assert fail_rotor > 0 and fail_rotor <=4, 'Failed rotor must be {1, 2, 3, 4}'
        failed_ids = [(fail_rotor -1)%4, (fail_rotor + 1) % 4]
        sum_forces = self.mass * self.gravity
        forces_halv = sum_forces/2.0 # forces in each safe rotor
        wabs = np.sqrt(forces_halv/self.kf)
        r_bar_initial = 2 * forces_halv * self.kt/self.dumping * 0.8 #TODO to avoid global minimum during optimization
        
        ws = [wabs * float(idx not in failed_ids) * (-1)**(idx%2) for idx in range(4)]
        initial_forces = [forces_halv * float(idx not in failed_ids) for idx in range(4)]
        wb = [2, 2, r_bar_initial * (-1)**(fail_rotor%2)]
        
        reduced_axis = [0.0, 0.0, 1.0]
        epsilon = 1/r_bar_initial # nz/r_bar
        return [*reduced_axis, *initial_forces, sum_forces, epsilon, *wb, *ws]

    def solve_generic(
        self,
        fail_rotor: int,
        alpha: float,
    ):
        # reject wrong values:
        assert fail_rotor > 0 and fail_rotor <=4, 'Failed rotor must be {1, 2, 3, 4}'
        """
            x[0]: nx
            x[1]: ny
            x[2]: nz
            x[3]: f1
            x[4]: f2
            x[5]: f3
            x[6]: f4
            x[7]: sF
            x[8]: E
            x[9]: p
            x[10]: q
            x[11]: r
            x[12]: w1
            x[13]: w2
            x[14]: w3
            x[15]: w4
        """
        mass = self.mass
        gravity = self.gravity
        alpha = alpha
        kt = self.kt
        kf = self.kf
        dumping = self.dumping
        length_arm = self.length_arm
        izzt = self.izzt
        ixxt = self.ixxt
        izzp = self.izzp
        fail_rotor = fail_rotor
        # general equations, i.e independent from which rotor is failed
        def get_equations(failed_rotor: int, alpha_ratio):
            def system_dynamics(x):
                general_eqs = [
                    x[7] * x[2] - mass * gravity,   # from equation #1 ..reference in paper (18)
                    x[0] - x[8] * x[9],             # from equation #2 ..reference paper (16)
                    x[1] - x[8] * x[10],            # from equation #3
                    x[2] - x[8] * x[11],            # from equation #4
                    x[11] - (kf*kt/dumping) * (x[12]**2 - x[13]**2 + x[14]**2 - x[15]**2), # from equation #5 ..reference in paper (21)
                    kf * (x[13]**2 - x[15]**2) * length_arm - (izzt - ixxt) * x[10] * x[11] - izzp * x[9] * (x[12] + x[13] + x[14] + x[15]), # from equation #6 ..reference in paper (12)
                    kf * (x[14]**2 - x[12]**2) * length_arm + (izzt - ixxt) * x[9] * x[11] + izzp * x[9] * (x[12] + x[13] + x[14] + x[15]), # from equation #7 ..reference in paper (13)
                    x[3] - kf * x[12]**2,           # from equation #8 ..reference in paper (10)
                    x[4] - kf * x[13]**2,           # from equation #9   ""
                    x[5] - kf * x[14]**2,           # from equation #10  ""
                    x[6] - kf * x[15]**2,           # from equation #11  ""
                    x[7] - x[3] - x[4] - x[5] - x[6], # from equation #12, reference in paper (18-)
                    ##alt
                    x[0]**2 + x[1]**2 + x[2]**2 - 1
                ]
                if failed_rotor == 1:
                    particular_equations = [
                        x[12] - 0,
                        x[5] - alpha_ratio * x[4],
                        x[4] - x[6],
                    ]
                elif failed_rotor == 2:
                    particular_equations = [
                        x[13] - 0,
                        x[6] - alpha_ratio * x[3],
                        x[3] - x[5],
                        #x[9] - 0,
                    ]
                elif failed_rotor == 3:
                    particular_equations = [
                        x[14] - 0,
                        x[3] - alpha_ratio * x[4],
                        x[4] - x[6],
                        #x[10] - 0,
                    ]
                elif failed_rotor == 4:
                    particular_equations = [
                        x[15] - 0,
                        x[4] - alpha_ratio * x[3],
                        x[3] - x[5],
                        #x[9] - 0,
                    ]
                else:
                    raise ValueError("Rotors are available in index [1-4]")
                eqs = general_eqs + particular_equations
                
                return eqs
            
            return system_dynamics

        dynamics = get_equations(fail_rotor, alpha)
        initial_conditions = self.get_initial_conditions(fail_rotor)
        #initial_conditions = get_initial_conditions(mass, gravity, kt, kf, dumping, fail_rotor)
        #root = fsolve(dynamics, [0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 8.0, 0.05, 2, 2, -20, 400, 400, 400, 400])
        root = fsolve(dynamics, initial_conditions)
        evaluation = dynamics(root)
        return root, evaluation
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotor', '--failed-rotor', '-fr', dest='failed_rotor', type=int, default=4)
    parser.add_argument('--alpha', '-p', dest='alpha', type=float, default=0.5)
    
    args = parser.parse_args()
    failed_rotor = args.failed_rotor
    alpha_ratio = args.alpha

    izzp = 76.6875e-6
    ixxt = 7.0015e-3
    iyyt = 7.075e-3
    izzt = 12.0766875e-3
    ixxb = 7e-3
    mass = 0.716
    kf = 8.54858e-6
    kt = 0.016
    damping = 2.75e-3
    l_arm = 0.17
    w_max = 838
    gravity = 9.81
    up_time_rotor_dynamics = 0.0125
    sol = SolutionQuadrotorWrapper(
        mass=mass,
        gravity=gravity,
        kt=kt,
        kf=kf,
        dumping=damping,
        length_arm=l_arm,
        izzt=izzt,
        ixxt=ixxt,
        izzp=izzp,
        ixxb=ixxb
    )
    np.set_printoptions(precision=3)
    A, B = sol.get_control_matrixes(failed_rotor, alpha_ratio)
    Ae, Be = sol.get_extended_control_matrixes(failed_rotor, alpha_ratio, up_time_rotor_dynamics)
    print('='*20)
    print(f"A=\n{A}")
    print(f"B=\n{B}")
    print(f"Ae=\n{Ae}")
    print(f"Be=\n{Be}")
