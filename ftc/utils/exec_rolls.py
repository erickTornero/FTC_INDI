import os
from ftc.utils.logger import Logger
from wrapper.wrapper_crippled import QuadrotorEnvRos

# Define rolls function to inject failures at specific timestep
import numpy as np
from typing import Optional

experiment_config = {
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
        "nrounds"       :   2
    },
    "controllers": {
        'fault_free': 'HoverController',
        'fault_case': 'indi',
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

def rollouts(
    env: QuadrotorEnvRos,
    trajectory_info: dict,
    controllers_info: dict,
    failure_info: dict,
    nrolls: int=20,
    max_path_length: int=5000,
    save_paths: os.path=None,
    logger: Optional[Logger]=None,
):
    allow_inject = failure_info['allow']
    if allow_inject:
        pass
