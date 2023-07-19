"""
An script to test push failure while flying
"""

import os
import re
import json
import glob
import math
from ftc.utils.logger import Logger
from ftc.utils.exec_rolls import exec_rollouts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--roll-id", "-ri", dest="roll_id", type=int, required=False)
    parser.add_argument("--trajectory-type", "-tt", dest="trajectory_type", type=str, default="point")
    parser.add_argument("--not-early-stop", "-nes", dest="not_early_stop", action="store_true", default=False)
    parser.add_argument("--inject-failure", "-if", dest="inject_failure", action="store_true", default=False)
    parser.add_argument("--ftc-algorithm", "-ftca", dest="ftc_algorithm", type=str, default="indi")
    parser.add_argument("--only-fault-free", "-off", dest="only_fault_free", action="store_true", default=False)

    # euler initial distribution
    parser.add_argument("--euler-initial-std", "--euler-std", dest="euler_initial_std", type=float, default=34.37) # in degrees
    parser.add_argument("--euler-initial-mean", "--euler-mean", dest="euler_initial_mean", type=float, nargs="*", required=False) # in degrees
    parser.add_argument("--euler-initial-mean-x", "--euler-mean-x", dest="euler_initial_mean_x", type=float, default=0.0) # in degrees
    parser.add_argument("--euler-initial-mean-y", "--euler-mean-y", dest="euler_initial_mean_y", type=float, default=0.0) # in degrees
    parser.add_argument("--euler-initial-mean-z", "--euler-mean-z", dest="euler_initial_mean_z", type=float, default=0.0) # in degrees

    args = parser.parse_args()

    roll_id = args.roll_id
    trajectory_type = args.trajectory_type
    not_early_stop = args.not_early_stop
    allow_inject_failure = args.inject_failure
    ftc_algorithm: str = args.ftc_algorithm
    only_fault_free: str = args.only_fault_free
    assert ftc_algorithm.lower() in ["indi", "lqr"], f"unsupported ftc algorithm -> {ftc_algorithm}"

    # euler angle parse
    euler_initial_std = args.euler_initial_std
    euler_initial_mean = args.euler_initial_mean
    euler_initial_mean_x = args.euler_initial_mean_x
    euler_initial_mean_y = args.euler_initial_mean_y
    euler_initial_mean_z = args.euler_initial_mean_z

    if euler_initial_mean:
        assert len(euler_initial_mean) == 3, f"Euler initial mean must be len 3, got -> {euler_initial_mean}"
    else:
        euler_initial_mean = [
            euler_initial_mean_x,
            euler_initial_mean_y,
            euler_initial_mean_z,
        ]
    # convert from degs to rad
    euler_initial_mean = [math.pi * v/180 for v in euler_initial_mean]
    euler_initial_std = math.pi * euler_initial_std/180.0

    save_paths = './data/rolls62'

    if roll_id is None:
        folds = os.listdir("./data")
        max_candidate = 1
        for fold in folds:
            match = re.search(r"rolls(\d+)", fold)
            if match:
                candidate = int(match.groups()[0])
                if candidate > max_candidate: 
                    max_candidate = candidate
        roll_id = max_candidate
        print(f"Found candidate -> rolls{roll_id}")

    save_paths = f"./data/rolls{roll_id}"

    experiment_config = {
        "type": "online_failure",
        "max_path_length": 5000,
        "nrollouts": 20,
        "early_stop": not not_early_stop,
        "environment": {
            'args_init_distribution': {
                'max_radius': 5.99,#4.2,
                'max_ang_speed': 30,
                'max_radius_init': 2.0,
                'angle_rad_std': 0.0,
                'angle_mean': [0.1745, 0.0, 0.0],
                'angular_speed_mean': 0.0,
                'angular_speed_std': 0.0,
            },
            "rate": 1000,
            'state_space_names': [
                'rotation_matrix', 'position', 'euler', 'linear_vel', 'angular_vel'
            ],
        },
        
        "trajectory_args"   :   {
            "wave"          :   trajectory_type,
            "nrounds"       :   1,
            "z_bias"        :   6.0,
        },
        "controllers": {
            'fault_free': 'HoverController',
            'fault_case': ftc_algorithm,
        },
        "inject_failure"    :   {
            "allow": allow_inject_failure,
            "type": "push",  # "push" or "ornstein"
            "damaged_motor_index": 2, #allowed [0, 1, 2, 3]
            "push_failure_at": 12000,
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
        #assert len(files_paths)==0, 'The folder is busy {}, select another one'.format(save_paths)
        if len(files_paths) > 0:
            user_input = input(f"fold <rolls{roll_id}> busy, want to use roll-id -> <rolls{roll_id + 1}>? yes/no >\t")
            if user_input.lower() == "yes":
                roll_id += 1
                save_paths = f"./data/rolls{roll_id}"
                logger = Logger(save_paths, 'logtest.txt')
            else:
                raise FileExistsError(f"Folder <{save_paths}> already exists")

        if not os.path.exists(save_paths):
            os.makedirs(save_paths)
        
        #with open(os.path.join(save_paths, 'experiment_config.json'), 'w') as fp:
        #    json.dump(experiment_config, fp, indent=2)
    print("="* 20 + "ARGS" + "="*20)
    print("save_folder\t->\t", save_paths)
    print("trajectory_type ->\t", trajectory_type)
    print("nrollsouts\t->\t", experiment_config["nrollouts"])
    print("early_stop\t->\t", not not_early_stop)
    print("allow_inject_failure\t->\t", allow_inject_failure)
    print("ftc_algorithm\t->\t", ftc_algorithm)
    print("-\tinitial distribution\t-")
    print(f"euler_initial_std -> {euler_initial_std}")
    print(f"euler_initial_mean -> {euler_initial_mean}")
    print(f"euler_initial_mean_x -> {euler_initial_mean_x}")
    print(f"euler_initial_mean_y -> {euler_initial_mean_y}")
    print(f"euler_initial_mean_z -> {euler_initial_mean_z}")
    print("="*50)

    exec_rollouts(
        environment_config=experiment_config['environment'],
        trajectory_info=experiment_config['trajectory_args'],
        controllers_info=experiment_config['controllers'],
        failure_info=experiment_config['inject_failure'],
        nrolls=experiment_config['nrollouts'],
        max_path_length=experiment_config['max_path_length'],
        save_paths=save_paths,
        logger=logger,
        run_all_steps=not experiment_config['early_stop'],
        experiment_config=experiment_config,
    )