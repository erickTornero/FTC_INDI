import rospy
class PID:
    """
        Proportional Integrative Derivative Controller
    """
    def __init__(self, kp, ki, kd):
        self.kp             =   kp
        self.ki             =   ki
        self.kd             =   kd
        self.integral_cum   =   0.0
        windup_guard_gain   =   500.0
        self.windup_guard   =   abs(windup_guard_gain/self.ki) if self.ki != 0.0 else windup_guard_gain
        self.last           =   0.0
        self.last_time      =   0.0

    def get_and_update_PID(self, target, current, dt):
        error               =   target - current
        proportional_term   =   self.kp*error
        self.integral_cum   +=  error * dt
        curr_time           =   rospy.get_time()
        elapsed_time        =   curr_time - self.last_time
        if elapsed_time > 1.0: elapsed_time = dt
        if elapsed_time < 1e-5: elapsed_time = dt
        #print(elapsed_time)
        if self.integral_cum > self.windup_guard:
            self.integral_cum = self.windup_guard
        elif self.integral_cum < -self.windup_guard:
            self.integral_cum -= self.windup_guard
        
        integral_term       =   self.ki * self.integral_cum
        derivative_term     =   self.kd * (current - self.last)/elapsed_time
        #if derivative_term != derivative_term:
        #    print('nan found dval: {} | elapsed time: {}'.format(derivative_term, elapsed_time))
        #    derivative_term = 0.0
        self.last           =   current
        self.last_time      =   curr_time
        return proportional_term + integral_term - derivative_term

    def init_timers(self):
        self.last_time      =   rospy.get_time()


class HoverPIDFF:
    def __init__(self):
        super().__init__()

class PIDv2:
    def __init__(self, kp, ki, kd):
        self.kp             =   kp
        self.ki             =   ki
        self.kd             =   kd
    
    def update(self, target, current):
        error   =   target - current
