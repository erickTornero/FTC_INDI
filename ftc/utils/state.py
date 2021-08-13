class State:
    def __init__(self, invert_axis=False):
        self.position = None
        self.quaternion = None
        self.linear_vel = None
        self.angular_vel = None
        self.rotation_matrix = None
        self.euler = None
        self.zTarget = None
        self.fail_id = None
        self.acc = None
        self._invert_axis = invert_axis

    def update(self, observation):
        self.position = self.invert(observation['position'])
        self.quaternion = observation['quaternion']
        self.linear_vel = self.invert(observation['linear_vel'])
        self.angular_vel = self.invert(observation['angular_vel'])
        self.rotation_matrix = observation['rotation_matrix']
        self.euler = self.invert(observation['euler'])
        self.acc = self.invert(observation['lin_acc'])
        self.w_speeds = observation['w_speeds']
    
    def update_fail_id(self, fail_id):
        self.fail_id = fail_id

    """
    Unnecessary for the momment id: 300

    def update_vel_ref(self, vel_ref):
        self.vel_ref = vel_ref
    
    def update_pos_ref(self, pos_ref):
        self.pos_ref = pos_ref
    """

    def invert(self, a):
        """
            Use just if you want to change the axis signs, this is a trick, make possible to no use it.
        """
        if self._invert_axis:
            if type(a)==tuple:
                a = list(a)
            a[2] = -a[2]
            a[1] = -a[1]
        return a
        

    @property
    def att(self):
        return self.euler

    @property
    def omegaf(self):
        #get anular speeds
        return self.angular_vel

    @property
    def pos(self):
        return self.position

    @property
    def vel(self):
        return self.linear_vel