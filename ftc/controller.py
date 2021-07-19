import time
import numpy as np
from ftc.functions import URPositionControl, URYawControl
from ftc import PseudoControllAttINDI, AllocationAttINDI 
from ftc.filters import LowpassFilter
from math import sin, cos

class INDIController:
    def __init__(self, parameters):
        self.errorInt = np.zeros(3, dtype=np.float32)
        self.parameters = parameters
        
        #Low pass Filters
        self.low_pass_ndes = LowpassFilter(1, parameters.t_indi)
        self.low_pass_zTarg = LowpassFilter(1, parameters.t_indi)
        self.low_pass_dY = LowpassFilter(1, parameters.t_indi)

        #Saturators
        self.saturator_w = Saturator(parameters.w_min, parameters.w_max)
        self.saturator_lambda = Saturator(-0.5, 0.5)

        #Derivators
        self.derivator_ndes = DiscreteTimeDerivative() ##WARN: Not specifying the period of sampling Ts 
        self.derivator_z = DiscreteTimeDerivative() ##WARN: Not specifying the period of sampling Ts
        self.derivator_dY = DiscreteTimeDerivative() ##WARN: Not specifying the period of sampling Ts
        
        self.pseudo_controll_att = PseudoControllAttINDI(parameters)
        self.allocation_att_indi = AllocationAttINDI(parameters)
        self.subsystem = Subsystem(parameters)

    def __call__(self, state, inputs):
        n_des, r_cmd = self.outer_controller(state, inputs)
        w_cmd = self.inner_controller(state, n_des, r_cmd)
        return w_cmd

    def outer_controller(self, state, inputs):
        n_des, self.errorInt = URPositionControl(inputs, state, self.parameters, self.errorInt)
        r_cmd = URYawControl(inputs, state, self.parameters)
        return n_des, r_cmd

    def inner_controller(self, state, n_des, r_sp):
        # Subsystem
        Tc = time.time()
        ssres = self.subsystem(state, n_des)
        h0, posdd, U0, U1 = ssres['h0'], ssres['posdd'], ssres['U0'], ssres['U1']

        #pseudo controll att indi
        n_des_f = self.low_pass_ndes(n_des, Tc)
        _lambda = self.derivator_ndes(n_des_f, Tc)
        _lambda = self.saturator_lambda(_lambda)

        nB = self._getnB(state.fail_id)
        Z_ref = state.zTarget
        Z_ref_f = self.low_pass_zTarg(Z_ref, Tc)
        Vz_ref = self.derivator_z(Z_ref_f, Tc)

        nu, dY, Y = self.pseudo_controll_att(state, n_des_f, _lambda, nB, r_sp, Z_ref, Vz_ref)

        ## Allocation Attitude Indi
        dY_f = self.low_pass_dY(dY, Tc)
        ddY = self.derivator_dY(dY_f, Tc)

        U, ddy0, dU = self.allocation_att_indi(state, nu, ddY, h0, posdd, U0, U1)

        w_cmd = self._U2Omega(U)
        w_cmd = self.saturator_w(w_cmd)
        return w_cmd

    def _U2Omega(self, U):
        return np.sqrt(np.abs(U)) * np.sign(U)

    def _getnB(self, fail_id):
        """inner loop, get nB"""
        a = self.parameters.axis_tilt
        if self.parameters.DRF_enable == 0 or self.parameters.DRF_enable == 1:
            if fail_id == 0:
                n = [-a, a, -1]
            elif fail_id == 1:
                n = [-a, -a, -1]
            elif fail_id == 2:
                n = [a, -a, -1]
            elif fail_id == 3:
                n = [ a, a, -1]
            else:
                n = [0, 0, -1]
        else:
            n = [0, 0, -1]

        return n


class Subsystem:
    def __init__(self, parameters):
        self.parameters = parameters
        self.lowpass_H = LowpassFilter(1, parameters.t_indi)
        self.lowpass_az = LowpassFilter(1, parameters.t_indi)
        self.lowpass_U0 = LowpassFilter(1, parameters.t_indi)
        self.lowpass_U1 = LowpassFilter(1, parameters.t_indi)

    def __call__(self, states, n_des):
        Ts = time.time()
        h = self._Hestimator(n_des, states.att)
        h0 = self.lowpass_H(h, Ts)
        aZ_filtered = self.lowpass_az(states.acc[2])
        posdd2 = self._posdd2Estimator(aZ_filtered, states.att)
        U_mea = self._omega2U(states.w_speeds)
        U0 = self.lowpass_U0(U_mea, Ts)
        U1 = self.lowpass_U1(states.omegaf[2], Ts)

        return {
            'h0': h0,
            'posdd': posdd2,
            'U0': U0,
            'U1': U1
        }

    def _Hestimator(self, n_des, att):
        phi = att[0]
        theta = att[1]
        psi = att[2]

        R_IB = np.array([
            [cos(psi) * cos(theta), cos(psi) * sin(theta) * sin(phi) - sin(psi) * cos(phi), cos(psi) * sin(theta) * cos(phi) + sin(psi) * sin(phi)],
            [sin(psi) * cos(theta), sin(psi) * sin(theta) * sin(phi) + cos(psi) * cos(phi), sin(psi) * sin(theta) * cos(phi) - cos(psi) * sin(phi)],
            [-sin(theta), cos(theta) * sin(phi), cos(theta) * cos(phi)]
        ])

        h = np.linalg.lstsq(R_IB, n_des)
        return h
    
    def _posdd2Estimator(self, az, att):
        phi = att[0]
        theta = att[1]
        zdd = az * cos(theta) * cos(phi) + 9.8124
        return zdd
    

    def _omega2U(self, w_speeds):
        U  = np.zeros(4)
        U = w_speeds ** 2.0
        return U

class Saturator:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, value):
        return np.clip(value, self.min_val, self.max_val)

class DiscreteTimeDerivative:
    def __init__(self, T_sampling=None):
        """
            Discrete time derivative
            y(tn) = K * (u(tn) - u(tn-1))/T
            T: Period of sampling
            K: Gain
            u(tn): signal
            u(tn-1): previous time signal
            y(tn): Derivative of the signal
        """
        self.T_prev = 0
        self.V_prev = 0
        self.T_sampling = T_sampling

    
    def start(self, value=0, Tc=0):
        self.T_prev = Tc
        self.V_prev = value
    
    def __call__(self, value, Tc=None):
        """
            Return the derivative of the signal
            @value: value of the signal
            @Tc: optional arg, if you not specify it. You must provide T_sampling in constructor
        """
        if Tc is None:
            if self.T_sampling is not None: 
                signalD = (value - self.V_prev)/self.T_sampling
            else:
                raise ValueError("""You must specidy the Period of sampling Ts, or give the
                                    Current time TC""")
        else:
            dT = Tc - self.T_prev
            signalD = (value - self.V_prev)/dT
            self.T_prev = Tc
        
        self.V_prev = signalD
        return signalD


class PseudoAttINDIControl:
    def __init__(self, parameters):
        pass

    def __call__(self, state, n_des, state1, r_sp, inputs):
        # Prepare inputs
        pass


class AllocationAttINDIControl:
    def __init__(self, parameters):
        pass

    def __call__(self):
        pass