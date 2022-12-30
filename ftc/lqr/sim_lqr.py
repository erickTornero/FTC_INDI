import numpy as np
from typing import Tuple

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

def solve(mass, g, p, kt, kf, dump, l_arm, izzt, ixxt, izzp):
    from scipy.optimize import fsolve
    def func(x):
        return [
            x[0] * x[1] * (2 + p) - mass * g,
            x[2] - (kt * kf * (2 - p)/dump) * x[3]**2,
            x[0] - kf * x[3]**2,
            x[4] - x[8] * x[6],
            x[5] - x[8] * x[7],
            x[1] - x[8] * x[2],
            x[4]**2 + x[5]**2 + x[1]**2 - 1,
            x[6],
            #x[4],
            p * x[0] * l_arm - (izzt - ixxt) * x[7] * x[2] - izzp * x[7] * (2 + np.sqrt(p)) * x[3],
        ]
    root = fsolve(func, [2.0, 0.9, 20, 500, 0.0, 0.0, 0.0, 2.0, 0.02])
    return root, func(root)
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
    args = parser.parse_args()
    p = args.p
    print("using p={:.3f}".format(p))

    get_params(p, nz, mass, gravity, failed_id, kf, kt, damping)

    root, cl = solve(mass, gravity, p, kt, kf, damping, l_arm, izzt, ixxt, izzp)
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
            izzp
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