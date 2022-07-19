import numpy as np

class Trajectory:
    """
        Generate trajectories for quadrotor
        Availabel trajectories:
        * sin-vertical
        * circle
        * helicoid
        * stepped
        * point
    """
    def __init__(self, npoints: int, z_bias: float=4.0):
        """
            Generate trajectories for quadrotor
            Availabel trajectories:
            * sin-vertical
            * circle
            * helicoid
            * stepped
            * point

            args:
            @npoints:  Number of points to generate in a trajectory (keep in mind max_path_length)
            @z_bias:   Generate the trajectory with a bias in z-axis (avoid to crash with ground before early stop)
        """

        self.dt         =   float(1.0/npoints)
        #self.wave       =   wave
        self.npoints    =   npoints
        #self.nrounds    =   nrounds
        self.z_bias     =   z_bias

        #assert nrounds >= 0, 'nrounds must be a possitive value'

    def gen_points(self, wave: str, nrounds: int) -> np.ndarray:
        """
            Generate a trajectory in numpy.ndarray format
            @wave: type of wave, sin-vertical, circle, helicoid, stepped, point, pulse
            @nrounds: How many rounds take, used for cycling waves like circle,
        """
        t   =   np.arange(self.npoints+1)
        if wave == 'sin-vertical':
            x   =   t * self.dt
            y   =   t * 0.0
            z   =   np.sin(2*np.pi*t*self.dt*nrounds) + self.z_bias

        elif wave   ==  'circle':
            x   =   np.cos(2 * np.pi * t * self.dt * nrounds)
            y   =   np.sin(2 * np.pi * t * self.dt * nrounds)
            z   =   np.ones_like(t) * self.z_bias
        
        elif wave  ==  'helicoid':
            """ Elicoid trajectory, two rounds"""
            sign = self.z_bias/abs(self.z_bias) if self.z_bias != 0 else 1
            x   =   np.cos(2*np.pi*t*self.dt*nrounds)
            y   =   np.sin(2*np.pi*t*self.dt*nrounds)
            z   =   nrounds * (t/self.npoints) * sign + self.z_bias
            
        elif wave  ==  'stepped':
            """ An stepped Trajectory will be generated"""
            x   =   np.ones_like(t, dtype=np.float32) * 0.8
            y   =   np.ones_like(t, dtype=np.float32) * 0.8
            z   =   np.ones_like(t, dtype=np.float32) * self.z_bias + 0.8
            i_step  =   (self.npoints * 12)//25   
            x[i_step:]  =   0.0
            y[i_step:]  =   0.0
            z[i_step:]  =   self.z_bias

        elif wave == 'point':
            """ Fixed point trajectory (0,0,0) """
            x   =   np.zeros_like(t, dtype=np.float32)
            y   =   np.zeros_like(t, dtype=np.float32)
            z   =   self.z_bias * np.ones_like(t, dtype=np.float32)
        
        elif wave == 'pulse':
            init_pulse  =   self.npoints // 3
            end_pulse   =   (self.npoints * 2) // 3
            x   =   np.ones_like(t, dtype=np.float32) * 0.8
            y   =   np.ones_like(t, dtype=np.float32) * 0.8
            z   =   np.ones_like(t, dtype=np.float32) * self.z_bias + 0.8
               
            x[init_pulse:end_pulse]  =   0.0
            y[init_pulse:end_pulse]  =   0.0
            z[init_pulse:end_pulse]  =   self.z_bias

        else:
            assert False, 'Trajectory not defined'
        
        x       =   x.reshape(-1,1)
        y       =   y.reshape(-1,1)
        z       =   z.reshape(-1,1)

        position    =   np.concatenate((x,y), axis=1)
        position    =   np.concatenate((position, z), axis=1)
        return position
        