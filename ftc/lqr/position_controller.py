"""
    Slow position controller
"""
import rospy
from geometry_msgs.msg import Vector3
import numpy as np
from typing import Tuple
from ftc.utils.state import State
from ftc.utils.inputs import Inputs

class PositionController:
    def __init__(
        self,
        tick_duration: float,   # in seconds
        nticks_activation: int, # int nticks to call calculation
        gravity: float,         # gravity, positive value
        kp_pos: np.ndarray,     # kp-pos gain pid control
        kp_vel: np.ndarray,     # kp-vel gain pid control
        ki_vel: np.ndarray,     # ki-vel gain pid control
        max_angle: float,       # max angle respect the parallel to gravity
        max_integral_error: float, # max integral, to avoid huge errors
        max_linear_speed: float,    # maximum linear speed allowed
    ) -> None:
        # initi
        self.gravity = np.abs(gravity)
        self.tick_duration = tick_duration
        self.nticks_activation = nticks_activation
        self.period_controller = nticks_activation * self.tick_duration
        # control params
        self.kp_pos = kp_pos
        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        # lims vars
        self.max_angle = np.abs(max_angle)
        self.max_linear_speed = np.abs(max_linear_speed)
        self.max_integral_error = np.abs(max_integral_error)
        self.initialize()
        self.n_des_publisher = rospy.Publisher(
            '/lqrcontroller/ndes',
            Vector3,
            queue_size=10
        )

    def initialize(self):
        self.max_lateral_force = np.abs(self.gravity * np.tan(self.max_angle))
        self.integral_error = np.zeros(3, dtype=np.float32)
        self.tick_counter = 0
        self.n_des = np.array([0, 0, 1], dtype=np.float32) #initial ndes

    def __call__(
        self,
        state: State,
        inputs: Inputs,
    ) -> np.ndarray:
        if self.tick_counter % self.nticks_activation == 0:
            self.n_des = self.calculate_ndes(state, inputs)
        self.tick_counter += 1
        return self.n_des

    def calculate_ndes(
        self,
        state: State,
        inputs: Inputs
    ) -> Tuple[np.ndarray, float]:
        pos_error = np.array([inputs.x_target, inputs.y_target, inputs.z_target] - state.pos)
        vel_target = self.kp_pos * pos_error
        vel_target_clipped = np.clip(vel_target, -self.max_linear_speed, self.max_linear_speed)
        vel_error = vel_target_clipped - state.vel
        integral_error = self.integral_error + vel_error * self.period_controller
        self.integral_error = np.clip(integral_error, -self.max_integral_error, self.max_integral_error)
        a_ref = self.kp_vel * vel_error + self.ki_vel * self.integral_error
        a_ref[2] = a_ref[2] + self.gravity
        lateral_ratio = np.sqrt(a_ref[0]**2 + a_ref[1]**2)/self.max_lateral_force
        scaler = max(lateral_ratio, 1)
        a_ref[:2] = a_ref[:2]/scaler
        n_des = a_ref/np.linalg.norm(a_ref)
        # publish
        msg = Vector3(*n_des.tolist())
        self.n_des_publisher.publish(msg)
        return n_des
        