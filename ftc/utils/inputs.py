class Inputs:
    def __init__(self):
        self.xTarget = None
        self.yTarget = None
        self.zTarget = None
        self.yawTarget = None

    def update(self, observation):
        pass
    
    def update_xtarget(self, x_target):
        self.xTarget = x_target
    
    def update_ytarget(self, y_target):
        self.yTarget = y_target

    def update_ztarget(self, z_target):
        self.zTarget = z_target
    
    def updatePositionTarget(self, position):
        self.update_xtarget(position[0])
        self.update_ytarget(position[1])
        self.update_ztarget(position[2])

    def update_yawTarget(self, yaw_target):
        self.yawTarget = yaw_target