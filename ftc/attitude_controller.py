import numpy as np
from scipy.linalg import block_diag

class AttitudeController:
    def __init__(self, ix, iy, iz, b, l, chi, mass, k0, t0):
        self.ix = ix
        self.iy = iy
        self.iz = iz
        self.b = b
        self.l = l
        self.chi = chi
        self.mass = mass

        self.k = k0
        self.t = t0
        self.beta = np.atan(b/l)

    def update(self, euler, h0):
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

        h1 = h0[0], h2 = h0[1], h3 = h0[2]

        G0 = np.array([
            -self.k/self.mass*np.cos(theta)*np.cos(phi)*np.ones(4), 
            -h3*Gq + h2*Gr,
            h3*Gp - h1*Gr,
            Gr
        ])

        R = block_diag(1, np.array([[np.cos(self.chi), np.sin(self.chi)], [-np.sin(self.chi), np.cos(self.chi)]]), 1)


