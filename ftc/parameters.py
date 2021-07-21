class Parameters:
    def __init__(self, parameters=None):
        if parameters is None:
            # Controller related Parameters
            # Controls
            self.freq = 500 # control frequency

            #
            #   (1)<-- b -->(2)
            #      \       / ^
            #       \     /  |
            #       /     \  | l
            #      /       \ v
            #   (4)         (3)

            self.fail_id = [3]      # index of the failured propeller
            self.DRF_enable = 1     # failure of two diagonal rotors?
            self.fail_time = 0.0    # moment failiure occurs

            # drone self
            self.b = 0.1150    # [m]
            self.l = 0.0875
            self.Ix = 0.0014    # [kg m^2]
            self.Iy = 0.0013
            self.Iz = 0.0025
            self.mass = 0.375   # [kg]
            self.g = 9.81

            self.k0 = 1.9e-6    # propeller thrust coefficient
            self.t0 = 1.9e-8    # torque coefficient
            self.w_max = 838   # max / min propeller rotation rates, [rad/s]
            self.w_min = 0

            # INDI reduced att control
            self.chi = 105          # output scheduling parameter, [deg].
            self.pos_z_p_gain = 5   # altitude control pd gains
            self.pos_z_d_gain = 3
            self.axis_tilt = 0.0    # primary axis tilting param, 0 ~ 0.2,  
                                    # must be 0 for double rotor failure cases

            self.att_p_gain = 200   # attitude control pd gains 
            self.att_d_gain = 30
            self.t_indi = 0.02      # low-pass filter time constant, [s]

            # Yaw control
            self.YRC_Kp_r = 5.0
            self.YRC_Kp_psi = 5.0

            # position control
            self.position_maxAngle = 30/57.3    # maximum thrust tilt angle [rad]
            self.position_Kp_pos = [1.5, 1.5, 1.5]  # position control gains
            self.position_maxVel = 10           # maximum velocity
            self.position_intLim = 5.0 
            self.position_Ki_vel = [1.0, 1.0, 1.0]  # velocity gains
            self.position_Kp_vel = [2.0, 2.0, 2.0]
        else:
            # Controller related Parameters
            # Controls
            self.freq = parameters['freq'] # control frequency

            self.fail_id =  parameters['fail_id']
            self.DRF_enable =  parameters['DRF_enable']
            self.fail_time =  parameters['fail_time']

            # drone self
            self.b = parameters['b']    # [m]
            self.l = parameters['l']
            self.Ix = parameters['Ix']    # [kg m^2]
            self.Iy = parameters['Iy']
            self.Iz = parameters['Iz']
            self.mass = parameters['mass']   # [kg]
            self.g = parameters['g']

            self.k0 = parameters['k0']    # propeller thrust coefficient
            self.t0 = parameters['t0']    # torque coefficient
            self.w_max = parameters['w_max']   # max / min propeller rotation rates, [rad/s]
            self.w_min = parameters['w_min']

            # INDI reduced att control
            self.chi = parameters['chi']          # output scheduling parameter, [deg].
            self.pos_z_p_gain = parameters['pos_z_p_gain']   # altitude control pd gains
            self.pos_z_d_gain = parameters['pos_z_d_gain']
            self.axis_tilt = parameters['axis_tilt']    # primary axis tilting param, 0 ~ 0.2,  
                                    # must be 0 for double rotor failure cases

            self.att_p_gain = parameters['att_p_gain']   # attitude control pd gains 
            self.att_d_gain = parameters['att_d_gain']
            self.t_indi = parameters['t_indi']      # low-pass filter time constant, [s]

            # Yaw control
            self.YRC_Kp_r = parameters['YRC_Kp_psi']
            self.YRC_Kp_psi = parameters['YRC_Kp_psi']

            # position control
            self.position_maxAngle = parameters['position_maxAngle']    # maximum thrust tilt angle [rad]
            self.position_Kp_pos = parameters['position_Kp_pos']  # position control gains
            self.position_maxVel = parameters['position_maxVel']          # maximum velocity
            self.position_intLim = parameters['position_intLim'] 
            self.position_Ki_vel = parameters['position_Ki_vel']  # velocity gains
            self.position_Kp_vel = parameters['position_Kp_vel']
    
    @property
    def gravity(self):
        return self.g
        
    """
    @property
    def freq(self):
        return self.freq
    
    @property
    def fail_id(self):
        return self.fail_id

    @property
    def DRF_enable(self):
        return self.DRF_enable # failure of two diagonal rotors?
    
    @property
    def fail_time(self):
        return self.fail_time

    # drone self
    @property
    def b(self):
        self.b
    
    @property
    def l(self):
        self.l
    
    @property
    def Ix(self):
        self.Ix
    
    @property
    def Iy(self):
        self.Iy
    
    @property
    def Iz(self):
        self.Iz

    @property
    def mass(self):
        self.mass
    
    @property
    def g(self):
        self.g

    def k0(self):
        return self.k0
    
    def t0(self):
        self.t0 = 1.9e-8    # torque coefficient
    
    def w_max(self):
        self.w_max = 1200   # max / min propeller rotation rates, [rad/s]
    
    def w_min(self):
        return self.w_min

    # INDI reduced att control

    def chi(self):
        return self.chi

    def pos_z_p_gain(self):
        self.pos_z_p_gain = 5   # altitude control pd gains

    def pos_z_d_gain(self):
        self.pos_z_d_gain = 3
    
    def axis_tilt(self):
        self.axis_tilt = 0.0    # primary axis tilting param, 0 ~ 0.2,  
                            # must be 0 for double rotor failure cases
    def att_p_gain(self):
        self.att_p_gain = 200   # attitude control pd gains 
    
    def att_d_gain(self):
        self.att_d_gain

    def t_indi(self):
        self.t_indi = 0.02      # low-pass filter time constant, [s]

    # Yaw control
    def YRC_Kp_r(self):
        self.YRC_Kp_r = 5.0
    
    def YRC_Kp_psi(self):
        self.YRC_Kp_psi = 5.0

    # position control
    def position_maxAngle(self):
        self.position_maxAngle = 30/57.3    # maximum thrust tilt angle [rad]
    
    self.position_Kp_pos = [1.5, 1.5, 1.5]  # position control gains
    self.position_maxVel = 10           # maximum velocity
    self.position_intLim = 5.0 
    self.position_Ki_vel = [1.0, 1.0, 1.0]  # velocity gains
    self.position_Kp_vel = [2.0, 2.0, 2.0]

    """

