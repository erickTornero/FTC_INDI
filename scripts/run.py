import os
import numpy as np
from ftc.utils.gen_trajectories import Trajectory
from ftc.indi.parameters import Parameters
from ftc.indi.controller import INDIController
from wrapper import QuadrotorEnvRos, state_space
from ftc.utils.rolls import rollouts
from ftc.utils.logger import Logger

crippled_degree = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
state_space = ['rotation_matrix', 'euler', 'position', 'linear_vel', 'angular_vel']

args_init_distribution = {
        'max_radius': 10,
        'max_ang_speed': 30,
        'max_radius_init': 0,
        'angle_rad_std': 0.0,
        'angular_speed_mean': 0,
        'angular_speed_std': 0.0,
}

quadrotor_parms_path = 'params/quad_parameters.json'
control_parms_path = 'params/control_parameters_indi.json'
parameters = Parameters(quadrotor_parms_path, control_parms_path)
rate = parameters.freq
Ts = 1/rate
if parameters.fail_id>=0:
    crippled_degree[parameters.fail_id] = 0.0

env = QuadrotorEnvRos(np.zeros(3, dtype=np.float32), crippled_degree, state_space, rate, **args_init_distribution)

controller = INDIController(parameters=parameters, T_sampling=Ts)

max_path_length = 10000

cum_reward = 0
trajectory_manager = Trajectory(max_path_length, -3)
trajectory = trajectory_manager.gen_points('helicoid', 2)

inject_failure = {'allow': False}
save_paths = './data/rollouts3'
logger  =   Logger(save_paths, 'logtest.txt')
nrolls = 20
rollouts(
    env, 
    controller, 
    nrolls, 
    max_path_length, 
    save_paths, 
    trajectory,
    logger=logger, 
    inject_failure=inject_failure)