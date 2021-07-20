class State:
    def __init__(self):
        self.position = None
        self.quaternion = None
        self.linear_vel = None
        self.angular_vel = None
        self.rotation_matrix = None
        self.euler = None
        self.zTarget = None
        self.fail_id = None
        self.acc = None

    def update(self, observation, w_speeds):
        self.position = observation['position']
        self.quaternion = observation['quaternion']
        self.linear_vel = observation['linear_vel']
        self.angular_vel = observation['angular_vel']
        self.rotation_matrix = observation['rotation_matrix']
        self.euler = observation['euler']
        self.acc = observation['lin_acc']
        self.w_speeds = w_speeds

    def update_ztarget(self, z_target):
        self.zTarget = z_target
    
    def update_fail_id(self, fail_id):
        self.fail_id = fail_id

    def update_vel_ref(self, vel_ref):
        self.vel_ref = vel_ref
    
    def update_pos_ref(self, pos_ref):
        self.pos_ref = pos_ref

    @property
    def att(self):
        return self.euler

    @property
    def omegaf(self):
        return self.angular_vel

    @property
    def pos(self):
        return self.position

    @property
    def vel(self):
        return self.linear_vel

