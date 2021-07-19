class LowpassFilter:
    def __init__(self, K, T):
        self.K = K # Gain
        self.T = T # Filter time constant
        self.t_prev = 0

        self.x = 0.0

    def start(self, value0=0, t=0):
        self.x = self.K * value0
        self.t_prev = t

    def __call__(self, value, Tc):
        Ts = Tc - self.t_prev
        self.t_prev = Tc
        self.x = (1 - Ts/self.T) * self.x + self.K *(Ts/self.T) * value
        return self.x