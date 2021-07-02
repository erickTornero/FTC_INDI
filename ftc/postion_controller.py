"""
    Position Controller based on PID
"""
import numpy as np

class PositionController:
    def __init__(self, mass, chi, gravity, kpz, kdz, katt_p, katt_d, kpr):
        self.mass = mass
        self.chi = chi
        self.gravity = gravity

        # gains
        self.kpz = kpz
        self.kdz = kdz
        self.katt_p = katt_p
        self.katt_d = katt_d
        self.kpr = kpr

        self.velocity_ref = np.zeros(3)
        self.position_ref = np.zeros(3)
        self.angular_vel_ref = np.zeros(3)

    def update(
        self, 
        euler: np.ndarray, 
        angular_vel: np.ndarray, 
        n_des: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        lambd: np.ndarray
    )->np.ndarray:
        phi = euler[0]
        theta = euler[1]
        psi = euler[2]

        nx, ny = 0, 0

        p = angular_vel[0]
        q = angular_vel[1]
        r = angular_vel[2]

        Rib = np.array([
            [np.cos(psi) * np.cos(theta), np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi), np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)],
            [np.sin(psi) * np.cos(theta) , np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi), np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)],
            [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)]
        ])
        h = np.matmul(np.linalg.inv(Rib), n_des)

        h1, h2, h3 = h[0], h[1], h[2]

        Y = np.array([[h1 - nx], [h2 - ny]])

        dY = np.array([
            [-h3 * q + h2 * r + lambd[0]],
            [h3 * p + h1 * r + lambd[1]]
        ])

        Y = np.matmul(
            np.array([
                [np.cos(self.chi), np.sin(self.chi)], 
                [-np.sin(self.chi), np.cos(self.chi)]
                ]), Y)
        dY = np.matmul(
            np.array([
                [np.cos(self.chi), np.sin(self.chi)], 
                [-np.sin(self.chi), np.cos(self.chi)]
                ]), dY)

        nu1 = - self.kdz * (velocity[2] - self.velocity_ref[2]) - self.kpz * (position[2] - self.position_ref[2])
        nu2 =  - self.katt_d * dY[0] - self.katt_p * Y[0]
        nu3 =  - self.katt_d * dY[1] - self.katt_p * Y[1]
        nu4 =  - self.kpr * (r-self.angular_vel_ref[2])

        nu = np.array([
            [nu1], [nu2[0]], [nu3[0]], [nu4],
        ])

        return nu

if __name__ == '__main__':
    pc = PositionController(1, 1, 9.81, 0.1, 0.1, 0.1, 0.1, 0.1)
    
    response = pc.update(
        np.array([0, 0,0.8 ]),
        np.array([1, 0.1, 8 ]),
        np.array([0, 0, 1 ]),
        np.array([-0.5, 0.8, 1.2 ]),
        np.array([1, 1.5, 0.8 ]),
        np.array([1, 1, 1 ]),
    )
    
    print(response)
        

