import time
import numpy as np
from ftc.functions import URPositionControl, URYawControl
from ftc import PositionController, AttitudeController 
from ftc.filters import LowpassFilter
from math import sin, cos

class INDIController:
    def __init__(self, parameters):
        self.errorInt = np.zeros(3, dtype=np.float32)
        self.parameters = parameters
        self.pseudo_controll_att = PseudoAttINDIControl(parameters)
        self.allocation_att_indi = AllocationAttINDIControl(parameters)
        self.subsystem = Subsystem(parameters)

    def outer_controller(self, state, inputs):
        n_des, self.errorInt = URPositionControl(inputs, state, self.parameters, self.errorInt)
        r_cmd = URYawControl(inputs, state, self.parameters)
        return n_des, r_cmd

    def inner_controller(self):
        pass

    def _subsystem(self, n_des, states):
        pass


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