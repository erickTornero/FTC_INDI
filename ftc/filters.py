class LowpassFilter:
    def __init__(self, K, T):
        self.K = K # Gain
        self.T = T # Filter time constant

        self.x = 0.0

    def start(self, value0):
        self.x = self.K * value0

    def __call__(self, value, Ts):
        self.x = (1 - Ts/self.T) * self.x + self.K *(Ts/self.T) * value
        return self.x