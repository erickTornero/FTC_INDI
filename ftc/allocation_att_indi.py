import numpy as np
from math import sin, cos
from scipy.linalg import block_diag
from ftc.parameters import Parameters
class AllocationAttINDI:
    def __init__(self, parameters: Parameters):
        self.ix = parameters.Ix
        self.iy = parameters.Iy
        self.iz = parameters.Iz
        self.b = parameters.b
        self.l = parameters.l
        self.chi = parameters.chi/57.3
        self.mass = parameters.mass
        self.k = parameters.k0
        self.t = parameters.t0
        self.beta = np.arctan(self.b/self.l)
        self.double_rotor = True
        self.DRF_enable = parameters.DRF_enable

    def __call__(self, state, nu, ddY, h0, zdd, U0, rdot):
        fail_flag = state.fail_id
        phi = state.att[0]
        theta = state.att[1]
        chi = self.chi
        if state.fail_id == 1 or state.fail_id == 3:
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
        ddy0 = np.array([zdd, ddY, rdot])
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
        
        if fail_flag > 0:
            if self.DRF_enable == 1:
                G[:, fail_id] = np.zeros((4, len(fail_id)))
                ddy0[2:,:] = np.zeros((2, 1))
                G[2:, :] = np.zeros_like(G[2:4, :])
                nu[2:, :] = np.zeros((2, 1)) 
            else:
                G[:fail_id] = np.zeros((4, 1))
                ddy0[3] = 0
                G[3, :] = np.zeros_like(G[3, :])
                nu[3] = 0

        dU = np.matmul(np.linalg.pinv(G), nu - ddy0)

        Y = (nu - ddy0)
        U = U0 + dU
        if fail_flag > 0:
            U[fail_id] = 0.0
        return U, Y, dU
        


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
    ac = AllocationAttINDI(
        0.01, 0.01, 0.21, 0.3, 0.25, 0.8, 1.65, 0.1, 0.1
    )

    response = ac.update(
        np.array([0.1, 0.3, 0.01]),
        np.array([0.2, 0.8, 0.22]),
        3.4, 3.3, 2.1, 
        np.array([0.43, 0.223, 0.01, 0.81]).reshape(4,1),
        np.array([2, 2, 2, 2]).reshape(4, 1),
        [1]
    )

    print(response)