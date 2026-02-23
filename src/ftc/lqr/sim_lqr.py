import numpy as np
from typing import Tuple
from scipy.optimize import fsolve

def get_n(nz: float) -> np.ndarray:
    assert np.abs(nz) <= 1.0, "invalid value of nz"
    nx = 0 # hardcoded
    ny = np.sqrt(1 - nz**2 - nx**2)
    return np.array([nx, ny, nz])

def get_forces(
    mass: float,
    g: float,
    p: float,
    failed_id: int,
    nz: float
) -> np.ndarray:
    """
        Get total forces in vector given a failed rotor
    """
    safe_ids, opposite_safe = get_indexes(failed_id)
    f = (mass * g)/(nz * (2 + p))
    mask = np.zeros(4)
    mask[safe_ids] = 1
    mask[opposite_safe] = p
    forces = mask * f
    return forces

def get_w_speeds(kf: float, forces: np.ndarray) -> np.ndarray:
    w_squared = forces/kf
    return np.sqrt(w_squared)

def get_indexes(failed_id: int) -> Tuple[tuple, int, int]:
    safe_ids = [(failed_id -1)%4, (failed_id + 1) % 4]
    safe_ids.sort()
    opposite_safe = (failed_id + 2)%4
    return safe_ids, opposite_safe

def get_radius_rps(
    nz: float,
    g: float,
    wB: np.ndarray
):
    alpha = np.sqrt(1-nz**2)/nz
    beta = g/(np.sum(wB * wB))
    return alpha * beta

def get_period_tps(wB: np.ndarray):
    return 2*np.pi/(np.linalg.norm(wB))

def get_yaw_speed(
    kt: float,
    kf: float,
    damping: float,
    w_speeds: np.ndarray,
    #w1: float,
    #w2: float,
    #w3: float,
    #w4: float
):
    alpha = (kt * kf)/damping
    #beta = w1**2 - w2**2 + w3**2 - w4**2
    #beta = w**2
    beta = np.sum(w_speeds[0::2]**2 - w_speeds[1::2]**2)
    return alpha * beta

def get_params(p: float, nz: float, mass, gravity, failed_id, kf, kt, damping):
    n = get_n(nz)
    forces = get_forces(mass, gravity, p, failed_id, nz)
    w_speeds = get_w_speeds(kf, forces)
    yaw_speed = get_yaw_speed(kt, kf, damping, w_speeds)
    wb = get_wb(n, yaw_speed)
    radius = get_radius_rps(nz, gravity, wb)
    period = get_period_tps(radius)
    # txt
    forces_txt = "Forces: f0={:.2f}, f1={:.2f}, f2={:.2f}, f3={:.2f}\n".format(*forces.tolist())
    w_speeds_txt = "w_speeds: w0={:.2f}, w1={:.2f}, w2={:.2f}, w3={:.2f}\n".format(*w_speeds.tolist())
    wb_txt = "WB: wbx={:.2f}, wby={:.2f}, wbz={:.2f}, ||wB||={}\n".format(*wb.tolist(), np.linalg.norm(wb))
    print(f"{forces_txt}{w_speeds_txt}{wb_txt}\nRadius: {radius}\nPeriod: {period}")


def get_wb(n: np.ndarray, yaw_speed: float) -> np.ndarray:
    nx, ny, nz = n[0], n[1], n[2]
    wbx = yaw_speed * nx/nz
    wby = yaw_speed * ny/nz
    return np.array([wbx, wby, yaw_speed])

def solve_generic(
    *,
    mass: float,
    gravity: float,
    alpha: float,
    kt: float,
    kf: float,
    dumping: float,
    length_arm: float,
    izzt: float,
    ixxt: float,
    izzp: float,
    fail_rotor: int
):
    
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
                x[6] - kf * x[16]**2,           # from equation #11  ""
                x[7] - x[3] - x[4] - x[5] - x[6], # from equation #12, reference in paper (18-)
            ]
            if failed_rotor == 1:
                particular_equations = [

                ]
            elif failed_rotor == 2:
                pass
            elif failed_rotor == 3:
                pass
            elif failed_rotor == 4:
                particular_equations = [
                    x[15] - 0,
                    x[4] - alpha_ratio * x[3],
                    x[3] - x[5],
                    x[9] - 0,
                ]
            else:
                raise ValueError("Rotors are available in index [1-4]")
            eqs = general_eqs + particular_equations
            
            return eqs
        return system_dynamics

    dynamics = get_equations(fail_rotor, alpha)
    root = fsolve(dynamics, [0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 8.0, 0.05, 2, 2, 20, 400, 400, 400, 400])
    evaluation = dynamics(root)
    return root, evaluation

def solve(mass, g, p, kt, kf, dump, l_arm, izzt, ixxt, izzp, fail_rotor=4):
    """
        Fail rotor from [1 to 4]
        We have 16 variables
    """
    from scipy.optimize import fsolve
    """
    Variables
        x[0]: 
        x[1]: nz
        x[2]: r_bar
        x[3]: wi // wi == (wi+2), wi+1 = p*wi # correspond to a failed-free rotor
        x[4]: nx_bar 
        x[5]: ny_bar
        x[6]: p_bar
        x[7]: q_bar
        x[8]: e, from equation 16 where n_bar = e * [p_bar, q_bar, r_bar]
    """

    def func_rotor_4(x):
        """
        if rotor 4 fails
        """
        return [
            x[0] * x[1] * (2 + p) - mass * g,           # .. from equation (18)
            x[2] - (kt * kf * (2 - p)/dump) * x[3]**2,  # ...from equation (21), this depends on which rotor fails (2 - p) for [w2 or w4] and (p-2) for [w1 or w4] failed
            x[0] - kf * x[3]**2,                        #from equation (10) fi = kf *wi^2
            x[4] - x[8] * x[6],                         #from equation (16), nx = e * p_bar
            x[5] - x[8] * x[7],                         #from equation (16), ny = e * q_bar
            x[1] - x[8] * x[2],                         #from equation (16), nz = e * r_bar
            x[4]**2 + x[5]**2 + x[1]**2 - 1,            #from equation (17) .. independent from which rotor fails
            x[6], #assumption: p_bar=0
            #x[4], ...#TODO: fix x0
            #p * x[0] * l_arm - (izzt - ixxt) * x[7] * x[2] - izzp * x[7] * (2 + np.sqrt(p)) * x[3],
            p * x[0] * l_arm - (izzt - ixxt) * x[7] * x[2] - izzp * x[7] * (2 + np.sqrt(p)) * x[3], # from equation 12
        ]

    def func_rotor_3(x):
        """
        if rotor 3 fails
        """
        return [
            x[0] * x[1] * (2 + p) - mass * g,
            #x[2] - (kt * kf * (p - 2)/dump) * x[3]**2,
            x[2] - (kt * (p - 2)/dump) * x[0],
            x[0] - kf * x[3]**2,
            x[4] - x[8] * x[6],
            x[5] - x[8] * x[7],
            x[1] - x[8] * x[2],
            #x[4]**2 + x[5]**2 + x[1]**2 - 1,
            np.linalg.norm(np.array([x[4], x[5], x[1]])) - 1,
            x[7],
            -p * x[0] * l_arm + (izzt - ixxt) * x[6] * x[2] + izzp * x[6] * (2 + np.sqrt(p)) * x[3],#from equation 13
        ]
    if fail_rotor == 4:
        root = fsolve(func_rotor_4, [2.0, 0.9, 20, 500, 0.0, 0.0, 0.0, 2.0, 0.02])
        res = func_rotor_4(root)
    elif fail_rotor == 3:
        root = fsolve(func_rotor_3, [2.0, 0.9, -20, 500, 0.0, 0.0, 0.0, 0.0, 0.02])
        res = func_rotor_4(root)
    else:
        raise NotImplementedError(f"rotor id {fail_rotor} not implemented yet")
    return root, res
if __name__ == "__main__":
    p = 0.655
    nz = 0.98
    mass = 0.5 # kg
    gravity = 9.81
    ixxt = 3.2e-3 # kgm2
    izzt = 5.5e-3 # kgm2
    izzp = 1.5e-5 # kgmm2
    ixxp = 0
    kt = 1.69e-2 # Nm
    kf = 6.41e-6 # Ns2/rad2
    failed_id = 3
    damping = 2.75e-3 # Nmsrad-1
    l_arm = 0.17
    ###
    izzp = 76.6875e-6
    ixxt = 7.0015e-3
    ixxb = 7.0e-3
    iyyt = 7.075e-3
    izzt = 12.0766875e-3
    mass = 0.716
    kf = 8.54858e-6
    kt = 0.016
    damping = 2.75e-3
    l_arm = 0.17
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', '-p', type=float, dest='p',default=0.0)
    parser.add_argument('--failed-rotor', '-fr', type=int, dest='failed_rotor', default=4)
    args = parser.parse_args()
    p = args.p
    failed_rotor = args.failed_rotor
    assert failed_rotor in [1, 2, 3, 4], 'Index of rotor not supported!'
    print("using p={:.3f}".format(p))

    get_params(p, nz, mass, gravity, failed_rotor, kf, kt, damping)

    root, cl = solve(mass, gravity, p, kt, kf, damping, l_arm, izzt, ixxt, izzp, failed_rotor)
    n = [root[4], root[5], root[1]]
    forces = [root[0], p * root[0], root[0], 0.0]
    wb = [root[6], root[7], root[2]]
    wbnorm = np.linalg.norm(np.array(wb))
    w_speeds = np.sqrt(np.array(forces)/kf).tolist()
    w1, w2, w3, w4 = w_speeds
    nx, ny, nz = root[4], root[5], root[1]
    rps = (np.sqrt(1 - nz**2)/nz) * gravity/(wbnorm**2) 
    tps = 2*np.pi/wbnorm
    r_ = wb[-1]
    a = ((ixxt - izzt)/ixxb)*r_ - (izzp/ixxb)*(w1+w2+w3+w4)
    print('='*30)
    print(f"using p={p:.2f}")
    print(f"nx={root[4]:.3f}, ny={root[5]:.3f}, nz={root[1]:.3f}")
    print(f"wx={root[6]:.2f}, wy={root[7]:.2f}, wz={root[2]:.2f}, ==> ||wB||={wbnorm:.2f}rad/s")
    print(f"f1={root[0]:.2f}N, f2={p * root[0]:.2f}N, f3={root[0]:.2f}N, f4={0.0:.2f}N")
    print(f"w1={w_speeds[0]:.2f}, w2={w_speeds[1]:.2f}, w3={w_speeds[2]:.2f}, w4={w_speeds[3]:.2f} rad/s")
    print(f"Rps: {rps*1000:.2f}mm")
    print(f"Tps: {tps:.2f}s")
    print(f"a_ = {a:.2f}, r = {r_:.2f}rad/s")
    #print(root)
    print(f"close to 0 => {np.sum(cl):.2f}")

    # ==================
    import matplotlib.pyplot as plt
    izzp = 76.6875e-6
    ixxt = 7.0015e-3
    iyyt = 7.075e-3
    izzt = 12.0766875e-3
    mass = 0.716
    kf = 8.54858e-6
    kt = 0.016
    damping = 2.75e-3
    l_arm = 0.17
    w_max = 838
    print("\n")
    print(f"max rotor ang speed: {w_max}rad/s")
    print(f"max force per rotor: {kf*w_max**2:.2f}N")
    print(f"max Torque Z per rotor: {kt*kf*w_max**2:.4f}Nm")
    print(f"weight: {mass*9.81:.3f}N")
    print("\n")

    fig, (axs_n, axs_wb) = plt.subplots(2, 1)
    ps = np.linspace(0.0, 4.0, 100)
    nxs, nys, nzs = [], [], []
    wbxs, wbys, wbzs = [], [], []
    for p in ps.tolist():
        root, cl = solve(
            mass, 
            gravity, 
            p, 
            kt, 
            kf, 
            damping, 
            l_arm, 
            izzt, 
            ixxt, 
            izzp,
            failed_rotor,
        )
        nx, ny, nz = root[4], root[5], root[1]
        wbx, wby, wbz = root[6], root[7], root[2]
        nxs.append(nx)
        nys.append(ny)
        nzs.append(nz)
        ##
        wbxs.append(wbx), wbys.append(wby), wbzs.append(wbz)
    axs_n.plot(ps, nxs)
    axs_n.plot(ps, nys)
    axs_n.plot(ps, nzs)
    axs_n.legend(['nx', 'ny', 'nz'])

    axs_wb.plot(ps, wbxs)
    axs_wb.plot(ps, wbys)
    axs_wb.plot(ps, wbzs)
    axs_wb.set_ylim(-35, 35)
    axs_wb.legend(['wbx', 'wby', 'wbz'])
    plt.show()
    

#with p=0.50
#using p=0.50
#nx=0.000, ny=0.040, nz=0.999
#wx=-0.00, wy=0.98, wz=24.54, ==> ||wB||=24.56rad/s
#f1=2.81N, f2=1.41N, f3=2.81N, f4=0.00N
#w1=573.52, w2=405.54, w3=573.52, w4=0.00 rad/s
#Rps: 0.65mm
#Tps: 0.26s
#a = ((ixxt - izzt)/ixxb)*r - (izzp/ixxb)*(w1+w2+w3+w4)