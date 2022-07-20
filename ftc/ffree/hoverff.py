import numpy as np

from wrapper.state_space import StateSpaceRobots
from .pid import PID
from .common import initialize_params
from ..base_controller import BaseController

def get_from_vector(vector, start, end):
    return vector[start:end]

def get_values(vector, indexes, state_space_names):
    return (get_from_vector(vector, **indexes[name]) for name in state_space_names)

class HoverController(BaseController):
    def __init__(self, state_space: StateSpaceRobots):
        self.ixx        =   0.007
        self.iyy        =   0.007
        self.izz        =   0.012
        self.m          =   0.68 + 4*0.009
        self.g          =   9.81        
        self.arm_length =   0.17
        self.force_c    =   8.54858e-06
        self.moment_c   =   0.016  #M_i = self.moment_c * force
        self.throtle    =   np.sqrt(self.m * self.g / (4 * self.force_c))
        self.pid_z      =   PID(1.0, 0.2, 3.0)
        self.pid_phi    =   PID(0.1, 0.2, 0.1)
        self.pid_roll   =   PID(0.1, 0.2, 0.1)
        self.pid_yaw    =   PID(0.1, 0.2, 0.1)
        self.pid_x      =   PID(0.003, 0.0, 10e-2)
        self.pid_y      =   PID(-0.003, -0.0, -10e-2)
        self.Td         =   0.0
        self.A          =   self.control_inputs_to_forces_rotors(self.arm_length, self.moment_c)
        print('Throtle {:.2f}'.format(self.throtle))
        self.position_gain  =   np.array([4.0, 4.0, 4.0], dtype=np.float32)
        self.velocity_gain  =   np.array([2.2, 2.2, 2.2], dtype=np.float32)
        self.angular_rate_gain  =   np.array([0.1, 0.1, 0.025], dtype=np.float32)
        self.attitude_gain  =   np.array([0.7, 0.7, 0.035], dtype=np.float32)
        self.mass       =   0.68 + 4*0.009
        self.gravity    =   9.81
        self.inertia_matrix =   np.array(
            [
                [7e-3, 0.0, 0.0], 
                [0.0, 7e-7, 0.0], 
                [0.0, 0.0, 12e-3]
            ], dtype=np.float32)
        self.rotors_configuration   =   self._init_rotors_conf('hummingbird')
        self.allocation_matrix, self.normalized_attitude_gain, self.normalized_angular_rate_gain, self.angular_acc_to_rotor_velocities = initialize_params(
            self.rotors_configuration, self.inertia_matrix, self.attitude_gain, self.angular_rate_gain
        )
        self.state_space_names = state_space.names 
        self.indexes_obs = state_space.get_state_space_indexes()

    def get_action(self, obs: np.ndarray, targetpos: np.ndarray, *args) -> np.ndarray:
        return self.compute_rotors_speeds_v2(*get_values(obs, self.indexes_obs, self.state_space_names), targetpos)

    def compute_rotors_speeds(self, rotmat, pos, euler, linear_vel, angular_vel):
        pos_error   =   self.target_p - pos
        vz          =   rotmat.reshape(3, 3).T[:, -1] 
        load_in_z   =   (self.m * self.g )/ np.dot(np.array([0.0, 0.0, 1]), vz)
        print(load_in_z)
        pid_force       =   self.pid_z.get_and_update_PID(self.target_p[2], pos[2], 5e-3) + load_in_z
        # Saturate total force about ~ 6N per propeller
        pid_force       =   min(pid_force, 18.0)
        abs_force       =   max(pid_force, 0.0)
        #pid_force       =   np.clip(pi)
        speeds          =   np.sqrt(abs_force/(4 * self.force_c))
        #speeds          =   
        total_speeds    =   min(speeds, 838.0)
        
        # pitch
        pid_x           =   self.pid_x.get_and_update_PID(0.0, pos[0], 5e-3)
        pid_y           =   self.pid_y.get_and_update_PID(0.0, pos[1], 5e-3)
        pid_roll        =   self.pid_roll.get_and_update_PID(pid_y, euler[0], 5e-3)
        pid_pitch       =   self.pid_phi.get_and_update_PID(pid_x, euler[1], 5e-3)
        pid_yaw         =   self.pid_yaw.get_and_update_PID(0.0, euler[2], 5e-3)
        #print('Forces: total| pitch| roll| yaw: {:-2f}| {:-2f}| {:-2f}| {:-2f}'.format(pid_force, pid_pitch, pid_roll, pid_yaw))
        #print('Pitch | roll | yaw: {:-2f}| {:-2f}| {:-2f}'.format(pid_pitch, pid_roll, pid_yaw))
            
        # position
        # x error
        #pid_x           =   self.pid_x.get_and_update_PID(0.0, pos[0], 5e-3)
        print('theta d: {:.2f}, {:.2f}'.format(pid_x, euler[1]))
        #pid_y           =   self.pid_y.get_and_update_PID(0.0, pos[1], 5e-3)
        #pid_pitch       +=  pid_x
        #pid_roll        +=  pid_y

        forces          =   np.matmul(self.A, np.array([pid_roll, pid_pitch, pid_yaw, pid_force], dtype=np.float32).T)
        angular_speeds  =   np.clip(np.sqrt(np.clip(forces, 0.0, np.inf)/self.force_c), 0.0, 838.0)

        #print(vel_body)
        #print('linear_vel {:.2f}, body vel {:.2f}'.format(linear_vel[0], vel_body[0]))
        #print(angular_speeds)        
        #print('Throtle + PID_speeds = force {:.2f} + {:.2f} = {:.2f} \terror {:.2f}'.format(self.throtle, speeds, total_speeds, pos_error[2]))
        return angular_speeds

    def control_inputs_to_forces_rotors(self, l, c):
        #forces2control  =   np.array([[0.0, -l, 0.0, l], [-l, 0.0, l, 0.0],[c, -c, c, -c], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        #forces2control  =   np.array([[0.0, l, 0.0, -l], [l, 0.0, -l, 0.0],[c, -c, c, -c], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        forces2control  =   np.array([[0.0, l, 0.0, -l], [-l, 0.0, l, 0.0],[c, -c, c, -c], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        return np.linalg.inv(forces2control)
    
    def compute_rotors_speeds_v2(
        self, rotmat: np.ndarray, 
        current_pos: np.ndarray, 
        euler: np.ndarray, 
        linvel: np.ndarray, 
        angvel: np.ndarray, 
        target_pos: np.ndarray
    ):
        position_error  =   current_pos - target_pos#current_pos #Wrapper outs: curr_pos - target_pos        
        acceleration    =   self.compute_desired_acceleration(position_error, rotmat, linvel)
        angular_acceleration        =   self.compute_desired_ang_acceleration(rotmat, angvel, acceleration)
        thrust                      =   -self.mass * np.dot(acceleration, rotmat.reshape(3,3)[:, 2])
        angular_accel_thrust        =   np.zeros(4, dtype=np.float32)
        angular_accel_thrust[:3]    =   angular_acceleration
        angular_accel_thrust[3]     =   thrust

        rotor_velocities    =   np.matmul(self.angular_acc_to_rotor_velocities, angular_accel_thrust)
        rotor_velocities    =   np.abs(rotor_velocities)
        rotor_velocities    =   np.sqrt(rotor_velocities)
        return rotor_velocities
        

    def compute_desired_acceleration(self, position_error, rotmat, linear_vel):
        rotmat          =   rotmat.reshape(3,3)
        target_vel      =   np.zeros(3, dtype=np.float32)
        target_acel     =   np.zeros(3, dtype=np.float32)  
        velocity_W      =   np.matmul(rotmat, linear_vel)
        velocity_error  =   velocity_W - target_vel
        #print(velocity_W)

        acceleration    =   ((position_error * self.position_gain) + (velocity_error * self.velocity_gain))/self.mass - np.array([0.0, 0.0, self.gravity]) - target_acel
        return acceleration


    def compute_desired_ang_acceleration(self, rotmat, angular_speed, acceleration):
        target_yaw      =   0.0
        target_yaw_rate =   0.0
        b1_des          =   np.array([np.cos(target_yaw), np.sin(target_yaw), 0.0])

        b3_des          =   -acceleration/np.linalg.norm(acceleration)
        b2_des          =   np.cross(b3_des, b1_des)
        b2_des          =   b2_des/np.linalg.norm(b2_des)

        rotmat_des      =   np.zeros((3,3), dtype=np.float32)
        rotmat_des[:, 0]    =   np.cross(b2_des, b3_des)
        rotmat_des[:, 1]    =   b2_des
        rotmat_des[:, 2]    =   b3_des

        if rotmat.ndim == 1:
            rotmat  =rotmat.reshape(3,3)

        angle_error_matrix  =   0.5 * (np.matmul(rotmat_des.T, rotmat) -np.matmul(rotmat.T, rotmat_des))
        angle_error         =   self.vector_from_skew_matrix(angle_error_matrix)

        angular_rate_des    =   np.zeros(3, dtype=np.float32)
        angular_rate_des[2] =   target_yaw_rate
        
        angular_rate_error  =   angular_speed - np.matmul(np.matmul(rotmat_des.T, rotmat), angular_rate_des)

        angular_acceleration    =   -1 * (angle_error * self.normalized_attitude_gain) - (angular_rate_error * self.normalized_angular_rate_gain) + np.cross(angular_speed, angular_speed)

        return angular_acceleration

    def vector_from_skew_matrix(self, skew_matrix):
        return np.array([skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]], dtype=np.float32)

    def _init_rotors_conf(self, quad_name='hummingbird'):
        if quad_name=='hummingbird':
            from collections import namedtuple
            rotors_conf =   namedtuple('RotorsConf', ['angle', 'arm_length', 'direction', 'rotor_force_constant','rotor_moment_constant'])
            rotors_configs =   [
                rotors_conf(0, 0.17, -1.0, 8.54858e-06, 0.016),
                rotors_conf(1.57079632679, 0.17, 1.0, 8.54858e-06, 0.016),
                rotors_conf(3.14159265359, 0.17, -1.0, 8.54858e-06, 0.016),
                rotors_conf(-1.57079632679, 0.17, 1.0, 8.54858e-06, 0.016)
            ]
            return rotors_configs
        else:
            raise NotImplementedError()