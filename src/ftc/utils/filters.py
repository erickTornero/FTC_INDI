from typing import Optional

from typing import Optional
class LowpassFilter:
    def __init__(self, K: float, T: float, T_sampling: Optional[float]=None):
        """
            Low pass filter
            
            @K: filter-gain
            @T: Filter time
            @T_sampling: optional
        """
        self.K = K # Gain
        self.T = T # Filter time constant
        self.t_prev = 0
        self.T_sampling = T_sampling # sampling frequency
        self.x = 0.0

    def start(self, value0: float=0, t: float=0):
        self.x = self.K * value0
        self.t_prev = t

    def __call__(self, value: float, Tc: float=None):
        if Tc is None:
            if self.T_sampling is None:
                raise ValueError("You must specidy the Period of sampling Ts, or give the Current time TC")
            Ts = self.T_sampling
        else:
            Ts = Tc - self.t_prev
        self.t_prev = Tc
        self.x = (1 - Ts/self.T) * self.x + self.K *(Ts/self.T) * value
        return self.x