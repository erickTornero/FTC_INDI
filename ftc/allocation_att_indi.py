import numpy as np
from math import sin, cos
from scipy.linalg import block_diag
from ftc.parameters import Parameters
class AllocationAttINDI:
    def __init__(self, parameters: Parameters):
        self.ix = parameters.Ix
        self.iy = parameters.Iy
        self.iz = parameters.Iz
        self.b = np.sqrt(parameters.b**2 + parameters.l**2)
        self.l = parameters.l
        self.chi = parameters.chi/57.3
        self.mass = parameters.mass
        self.k = parameters.k0
        self.t = parameters.t0
        self.beta = np.arctan(parameters.b/self.l)
        self.double_rotor = True
        self.DRF_enable = parameters.DRF_enable

    def __call__(self, state, nu, ddY, h0, zdd, U0, rdot):
        fail_flag = state.fail_id
        phi = state.att[0]
        theta = state.att[1]
        chi = self.chi
        if state.fail_id == 0 or state.fail_id == 2:
            chi = np.pi - self.chi

        h1 = h0[0]
        h2 = h0[1]
        h3 = h0[2]

        Gp = np.array([
            self.k * self.b * sin(self.beta), 
            -self.k * self.b * sin(self.beta), 
            -self.k * self.b * sin(self.beta), 
            self.k * self.b * sin(self.beta)
        ])/ self.ix

        Gq = np.array([
            self.k * self.b * np.cos(self.beta), 
            self.k * self.b * np.cos(self.beta), 
            -self.k * self.b * np.cos(self.beta), 
            -self.k * self.b * np.cos(self.beta)
        ]) / self.iy

        Gr = np.array([self.t, -self.t, self.t, -self.t]) / self.iz
        
        G0 = np.array([
            -self.k/self.mass * cos(theta) * cos(phi) * np.ones(4), 
            -h3 * Gq + h2 * Gr,
            h3 * Gp - h1 * Gr,
            Gr
        ])

        R = block_diag(1, np.array([[cos(chi), sin(chi)], [-sin(chi), cos(chi)]]), 1)
        ddy0 = np.vstack([zdd, ddY, rdot])
        G = np.matmul(R, G0)

        if self.DRF_enable and fail_flag >= 0:
            if fail_flag == 0 or fail_flag == 2:
                fail_id = [0, 2]
            elif fail_flag == 1 or fail_flag == 3:
                fail_id = [1, 3]
            else:
                raise NotImplementedError("")
        else:
            fail_id = fail_flag
        
        if fail_flag >= 0:
            if self.DRF_enable == 1:
                G[:, fail_id] = np.zeros((4, len(fail_id)))
                ddy0[2:,:] = np.zeros((2, 1))
                G[2:, :] = np.zeros_like(G[2:, :])
                nu[2:, :] = np.zeros((2, 1)) 
            else:
                G[:fail_id] = np.zeros((4, 1))
                ddy0[3,:] = 0  #assuming vector nx1
                G[3, :] = np.zeros_like(G[3, :])
                nu[3,:] = 0#assuming vector nx1
        try:
            dU = np.matmul(np.linalg.pinv(G), nu - ddy0)
        except np.linalg.LinAlgError as e:
            import pdb; pdb.set_trace()
            x = 21
        
        if (self._check_nan(dU)):
            import pdb; pdb.set_trace()
            x=22

        Y = (nu - ddy0)
        U = U0 + dU
        if fail_flag > 0:
            U[fail_id] = 0.0
        return U, Y, dU
    
    def _check_nan(self, v: np.ndarray):
        return np.isnan(v).any()
        


    def update(self, euler, h0, zdd, ddy, rdot, nu, U0, fail_indexes: list):
        phi = euler[0]
        theta = euler[1]

        Gp = np.array([
            self.k*self.b*np.sin(self.beta), 
            -self.k*self.b*np.sin(self.beta), 
            -self.k*self.b*np.sin(self.beta), 
            self.k*self.b*np.sin(self.beta)
        ])/ self.ix

        Gq = np.array([
            self.k*self.b*np.cos(self.beta), 
            self.k*self.b*np.cos(self.beta), 
            -self.k*self.b*np.cos(self.beta), 
            -self.k*self.b*np.cos(self.beta)
        ]) / self.iy

        Gr = np.array([self.t, -self.t, self.t, -self.t]) / self.iz

        h1 = h0[0]
        h2 = h0[1]
        h3 = h0[2]

        G0 = np.array([
            -self.k/self.mass*np.cos(theta)*np.cos(phi)*np.ones(4), 
            -h3*Gq + h2*Gr,
            h3*Gp - h1*Gr,
            Gr
        ])

        R = block_diag(1, np.array([[np.cos(self.chi), np.sin(self.chi)], [-np.sin(self.chi), np.cos(self.chi)]]), 1)

        ddy0 = np.array([[zdd], [ddy], [rdot], [0]])

        G = np.matmul(R, G0)

        ## if
        if self.double_rotor:
            for fail_index in fail_indexes:
                if fail_index in [0, 2]:
                    fail_id = [0, 2]
                else:
                    fail_id = [1, 3]
        else:
            fail_id = 1

        if self.double_rotor:
            G[:, fail_id] = np.zeros((4, len(fail_id)))
            ddy0[2:,:] = np.zeros((2, 1))
            G[2:, :] = np.zeros_like(G[2:3, :])
            nu[2:, :] = np.zeros((2, 1)) 
        else:
            G[:fail_id] = np.zeros((4, 1))
            ddy0[3] = 0
            G[3, :] = np.zeros_like(G[3, :])
            nu[3] = 0
        
        dU = np.matmul(np.linalg.pinv(G), nu - ddy0)

        Y = (nu - ddy0)
        U = U0 + dU
        return U, Y

if __name__ == '__main__':
    params = Parameters()
    ac = AllocationAttINDI(params)
    from ftc.state import State
    state = State()
    state.update({
        'position': None,
        'quaternion': None,
        'linear_vel': None,
        'angular_vel': None,
        'rotation_matrix': None,
        'euler': np.array([0.0030, -0.0017, -1.7837e-04]),
        'lin_acc': None,
        'w_speeds': None
    })

    state.update_fail_id(2)
    nu = np.array([-0.0037, 9.0707, 14.0776, 4.4383]).reshape(-1, 1)
    ddY = np.array([-46.2620, -28.0941]).reshape(-1, 1)
    h0 = np.array([9.9356e-05, 4.9287e-04, -1.0]).reshape(-1, 1)
    zdd = 8.3969
    U0 = np.array([2.4388e04, 2.4388e04, 2.4388e04, 2.6967e04]).reshape(-1, 1)
    rdot = -0.1227
    U, Y, dU = ac(state, nu, ddY, h0, zdd, U0, rdot)
    
    #expected
    # U = np.array([0, 1.0039e06, 0, 0.7055e06]).reshape(-1, 1)
    # Y = np.array([-8.4006, 55.3327, 0, 0]).reshape(-1, 1)
    # dU = np.array([0, 9.7959e05, 0, 6.7852e05]).reshape(-1, 1)
        