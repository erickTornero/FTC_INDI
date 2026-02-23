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
        #import pdb;pdb.set_trace()
        pos_error = inputs.position_target - state.pos
        vel_target = self.kp_pos * pos_error
        vel_target_clipped = np.clip(
            vel_target,
            -self.max_linear_speed,
            self.max_linear_speed
        )
        vel_error = vel_target_clipped - state.vel
        integral_error = self.integral_error + vel_error * self.period_controller
        self.integral_error = np.clip(
            integral_error,
            -self.max_integral_error,
            self.max_integral_error
        )
        a_ref = 2.0 * vel_target + self.kp_vel * vel_error + self.ki_vel * self.integral_error
        a_ref[2] = a_ref[2] + self.gravity
        lateral_ratio = np.sqrt(a_ref[0]**2 + a_ref[1]**2)/self.max_lateral_force
        scaler = max(lateral_ratio, 1)
        a_ref[:2] = a_ref[:2]/scaler
        n_des = a_ref/np.linalg.norm(a_ref)
        # publish
        msg = Vector3(*n_des.tolist())
        #self.n_des_publisher.publish(msg)
        return n_des

if __name__ == "__main__":
    nticks                  =   5
    gravity                 =   9.81    # m/s2
    frequency_controller    =   500     # hz

    kp_pos = np.array([1, 1, 1])
    kp_vel = np.array([1, 1, 1])
    ki_vel = np.array([1, 1, 1])
    max_angle = 0.523                   # 30 degrees in rads
    max_integral_error = 1.0
    max_linear_speed = 3.0

    pos_c = PositionController(
        1/frequency_controller,
        nticks,
        gravity,
        kp_pos,
        kp_vel,
        ki_vel,
        max_angle,
        max_integral_error,
        max_linear_speed
    )

    state = State()
    state.update({
        "position":         np.zeros(3, dtype=np.float32),  # important for position controller
        "linear_vel":       np.array([0.0, 1.0, 0.0]),      # important for position controller
        "quaternion":       None,
        "angular_vel":      None,
        "rotation_matrix":  None,
        "euler":            None,
        "lin_acc":          None,
        "w_speeds":         None,
    })

    inputs = Inputs()
    inputs.update_position_target([0, 1, 1])
    ndes = pos_c(state, inputs)
    print(ndes)