import json
from math import sqrt
import numpy as np
class Parameters:
    def __init__(self, quad_params_path=None, control_params_path=None):
        if quad_params_path is None or control_params_path is None:
            raise NotImplementedError("")
        else:
            parameters = self.load_params(quad_params_path, control_params_path)
            self.freq = parameters['freq'] # control frequency

            self.fail_id =  parameters['fail_id']

            self.b = parameters['body_width']    # b -> key [m]
            self.l = parameters['body_height']    # l -> key
            self.Ix = parameters['Ix']    # [kg m^2]
            self.Iy = parameters['Iy']
            self.Iz = parameters['Iz']
            self.mass = parameters['mass']   # [kg]
            self.g = parameters['gravity']   #key -> g

            self.k0 = parameters['motor_constant']    # k0 key -> propeller thrust coefficient
            self.t0 = parameters['moment_constant'] * self.k0    # t0 key -> torque coefficient
            self.w_max = parameters['w_max']   # max / min propeller rotation rates, [rad/s]
            self.w_min = parameters['w_min']

            #raise NotImplementedError("Load here your control parameters gven in control_params_path, if you dont need, please remove this line")
            self.Iu =   sqrt(self.Ix**2 + self.Iy**2)
            self.Iv =   sqrt(self.Ix**2 + self.Iy**2)
            self.s  =   sqrt(self.l**2 + self.b**2)
            self.Q  =   np.diag([0, 0, 2, 2, 0.0, 0.0])
            self.R  =   np.eye(2)
            self.Ip =   6e-6 # arbitrary, calculate this one
            self.act_dyn = 30
            #
            #self.K_lqr0 = get_lqr_matrix(par,0,0);
            #self.K_lqr1 = get_lqr_matrix(par,1,0);
            #self.K_lqr2 = get_lqr_matrix(par,2,0);
            #self.K_lqr3 = get_lqr_matrix(par,3,0);
            #self.K_lqr4 = get_lqr_matrix(par,4,0);
            #self.K_lqr13 = get_lqr_matrix(par,1,1);
            #self.K_lqr24 = get_lqr_matrix(par,2,1);

            self.k_lqrff            =   None
            self.k_lqr0             =   None
            self.k_lqr1             =   None
            self.k_lqr2             =   None
            self.k_lqr3             =   None
            
            self.k_lqr02            =   None
            self.k_lqr13            =   None
            self.DRF_enable         =   parameters['DRF_enable']
            self.axis_tilt          =   parameters['axis_tilt']
            self.YRC_Kp_r           =   parameters['YRC_Kp_r']
            self.YRC_Kp_psi         =   parameters['YRC_Kp_psi']

            # control params
            self.kpz_pos            =   parameters['kpz_pos']
            self.kdz_pos            =   parameters['kdz_pos']
            self.params             =   parameters

            # position control
            self.position_maxAngle  =   parameters['position_maxAngle']/57.3    # maximum thrust tilt angle [rad]
            self.position_Kp_pos    =   parameters['position_Kp_pos']  # position control gains
            self.position_maxVel    =   parameters['position_maxVel']          # maximum velocity
            self.position_intLim    =   parameters['position_intLim'] 
            self.position_Ki_vel    =   parameters['position_Ki_vel']  # velocity gains
            self.position_Kp_vel    =   parameters['position_Kp_vel']
            self.t_filter           =   parameters['t_filter']
            self.actuator_dynamics  =   parameters['time_constant_up']
            self.arm_length         =   parameters['arm_length']
            self.kt                 =   parameters['moment_constant']
            self.kf                 =   parameters['motor_constant']
            self.izzp               =   76.6875e-6
            self.ixxt               =   7.0015e-3
            self.iyyt               =   7.075e-3
            self.izzt               =   12.0766875e-3
            self.ixxb               =   7.0e-3
            self.aerodynamics_damping =   parameters['aerodynamics_damping']
            self.double_rotor       =   parameters['double_rotor']
            self.alpha_ratio        =   parameters['alpha_ratio'] # ratio of force of the counter rotor
            assert self.alpha_ratio >=0.0 and self.alpha_ratio < 1.0, f"alpha ratio must be in range [0, 1> but got {self.alpha_ratio}"

    @property
    def gravity(self):
        return self.g

    def load_params(self, quad_params_fp, control_params_fp):
        q_params = self._read_quad_params(quad_params_fp)
        c_params = self._read_control_params(control_params_fp)
        return {**q_params, **c_params}

    def _read_quad_params(self, quad_params_fp):
        return self._open_file(quad_params_fp)

    def _read_control_params(self, control_params_fp):
        return self._open_file(control_params_fp)

    def _open_file(self, file_path):
        with open(file_path, 'r') as fp:
            return json.load(fp)