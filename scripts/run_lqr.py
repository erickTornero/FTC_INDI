import os
import json
import glob
import numpy as np
from ftc.utils.gen_trajectories import Trajectory
from ftc.lqr.parameters import Parameters
from ftc.lqr.controller import LQRController
from wrapper import QuadrotorEnvRos, state_space
from ftc.utils.rolls import rollouts
from ftc.utils.logger import Logger

if __name__ == '__main__':
    crippled_degree = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    state_space = ['rotation_matrix', 'euler', 'position', 'linear_vel', 'angular_vel']
    save_paths = './data/rolls70'
    config = {
        "max_path_length": 10000,
        "nrollouts": 20,
        'args_init_distribution': {
            'max_radius': 4.2,
            'max_ang_speed': 30,
            'max_radius_init': 0.0,
            'angle_rad_std': 0.0,
            'angular_speed_mean': 0.0,
            'angular_speed_std': 0.0,
        },
        "trajectory_args"   :   {
            "wave"          :   'point',
            "nrounds"       :   2,
            "z_bias"        :   6.0,
        },
        "inject_failure"    :   {
            "allow": False,
            "type": "push",  # "push" or "ornstein"
            "damaged_motor_index": 2, #allowed [0, 1, 2, 3]
            "push_failure_at": 200,
            "push_new_task": 0.0,
            "ornstein_uhlenbeck": {
                "delta": 0.03,
                "sigma": 0.20, 
                "ou_a":  0.5,
                "ou_mu": 0.78, 
                "seed":  42
            }
        }
    }
    zbias = -6.0

    quadrotor_parms_path = 'params/quad_parameters.json'
    control_parms_path = 'params/control_params_lqr.json'
    parameters = Parameters(quadrotor_parms_path, control_parms_path)
    config['params'] = parameters.params

    rate = parameters.freq
    Ts = 1/rate
    if parameters.fail_id>=0:
        crippled_degree[parameters.fail_id] = 0.0

    

    max_path_length = config['max_path_length']

    trajectory_manager = Trajectory(max_path_length)
    trajectory = trajectory_manager.gen_points(**config['trajectory_args'])

    env = QuadrotorEnvRos(trajectory[0], crippled_degree, state_space, rate, **config['args_init_distribution'])

    controller = LQRController(parameters=parameters, T_sampling=Ts, state_space=env.state_space)

    inject_failure = config['inject_failure']
    
    logger  =   Logger(save_paths, 'logtest.txt')
    nrolls = config['nrollouts']

    if save_paths is not None:
        configsfiles    =   glob.glob(os.path.join(save_paths,'*.json'))
        files_paths     =   glob.glob(os.path.join(save_paths,'*.pkl'))

        assert len(configsfiles) ==0, 'The folder is busy, select other'
        assert len(files_paths)==0, 'The folder is busy, select another one'
        if not os.path.exists(save_paths):
            os.makedirs(save_paths)
        
        with open(os.path.join(save_paths, 'experiment_config.json'), 'w') as fp:
            json.dump(config, fp, indent=2)

    rollouts(
        env, 
        controller, 
        nrolls, 
        max_path_length, 
        save_paths, 
        trajectory,
        logger=logger,
        inject_failure=inject_failure,
        run_all_steps=True,
    )
