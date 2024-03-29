"""
An script to test push failure while flying
"""

import os
import json
import glob
from ftc.utils.logger import Logger
from ftc.utils.exec_rolls import exec_rollouts

if __name__ == "__main__":
    save_paths = './data/rolls62'

    experiment_config = {
        "max_path_length": 5000,
        "nrollouts": 20,
        "early_stop": True,
        "environment": {
            'args_init_distribution': {
                'max_radius': 6.0,#4.2,
                'max_ang_speed': 30,
                'max_radius_init': 2.0,
                'angle_rad_std': 0.0,
                'angular_speed_mean': 0.0,
                'angular_speed_std': 0.0,
            },
            "rate": 1000,
            'state_space_names': [
                'rotation_matrix', 'position', 'euler', 'linear_vel', 'angular_vel'
            ],
        },
        
        "trajectory_args"   :   {
            "wave"          :   'point',
            "nrounds"       :   1,
            "z_bias"        :   6.0,
        },
        "controllers": {
            'fault_free': 'HoverController',
            'fault_case': 'indi',
        },
        "inject_failure"    :   {
            "allow": False,
            "type": "push",  # "push" or "ornstein"
            "damaged_motor_index": 2, #allowed [0, 1, 2, 3]
            "push_failure_at": 2500,
            "push_new_task": 0.0,
            "delay_controller": 20,
            "ornstein_uhlenbeck": {
                "delta": 0.03,
                "sigma": 0.20, 
                "ou_a":  0.5,
                "ou_mu": 0.78, 
                "seed":  42
            }
        },
        "switch_info": [
            
        ]
    }

    logger  =   Logger(save_paths, 'logtest.txt')

    if save_paths is not None:
        configsfiles    =   glob.glob(os.path.join(save_paths,'*.json'))
        files_paths     =   glob.glob(os.path.join(save_paths,'*.pkl'))

        #assert len(configsfiles) ==0, 'The folder is busy, select other'
        assert len(files_paths)==0, 'The folder is busy {}, select another one'.format(save_paths)
        if not os.path.exists(save_paths):
            os.makedirs(save_paths)
        
        #with open(os.path.join(save_paths, 'experiment_config.json'), 'w') as fp:
        #    json.dump(experiment_config, fp, indent=2)

    exec_rollouts(
        environment_config=experiment_config['environment'], 
        trajectory_info=experiment_config['trajectory_args'], 
        controllers_info={}, 
        failure_info=experiment_config['inject_failure'],
        nrolls=experiment_config['nrollouts'],
        max_path_length=experiment_config['max_path_length'],
        save_paths=save_paths,
        logger=logger,
        run_all_steps=not experiment_config['early_stop'],
        experiment_config=experiment_config,
    )

