import numpy as np

def calculateAllocationMatrix(rotors_configuration):
    allocation_matrix   =   np.zeros((4,4), dtype=np.float32)
    for i, rotor in enumerate(rotors_configuration):
        allocation_matrix[0, i] = np.sin(rotor.angle) * rotor.arm_length * rotor.rotor_force_constant
        allocation_matrix[1, i] = -np.cos(rotor.angle) * rotor.arm_length * rotor.rotor_force_constant
        allocation_matrix[2, i] = -rotor.direction * rotor.rotor_force_constant * rotor.rotor_moment_constant
        allocation_matrix[3, i] = rotor.rotor_force_constant
    
    return allocation_matrix

def initialize_params(rotors_configuration, inertia_matrix, attitude_gain, angular_rate_gain):
    I4                              =   np.zeros((4,4), dtype=np.float32)
    I4[:3, :3]                      =   inertia_matrix
    I4[-1, -1]                      =   1.0   
    allocation_matrix               =   calculateAllocationMatrix(rotors_configuration)
    normalized_attitude_gain        =   np.matmul(attitude_gain.T, np.linalg.inv(inertia_matrix))
    normalized_angular_rate_gain    =   np.matmul(angular_rate_gain.T, np.linalg.inv(inertia_matrix))
    angular_acc_to_rotor_velocities =   np.matmul(np.matmul(allocation_matrix.T, np.linalg.inv(np.matmul(allocation_matrix, allocation_matrix.T))), I4)
    return allocation_matrix, normalized_attitude_gain, normalized_angular_rate_gain, angular_acc_to_rotor_velocities