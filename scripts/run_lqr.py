import os
import re
import json
import glob
import numpy as np
from ftc.utils.gen_trajectories import Trajectory
from ftc.lqr.parameters import Parameters
from ftc.lqr.controller import LQRController
from wrapper import QuadrotorEnvRos
from ftc.utils.rolls import rollouts
from ftc.utils.logger import Logger

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--roll-id", "-ri", dest="roll_id", type=int, required=False)
    parser.add_argument("--trajectory-type", "-tt", dest="trajectory_type", type=str, default="point")
    parser.add_argument("--not-early-stop", "-nes", dest="not_early_stop", action="store_true", default=False)
    args = parser.parse_args()
    roll_id = args.roll_id
    trajectory_type = args.trajectory_type
    not_early_stop = args.not_early_stop

    assert trajectory_type in ["point", "circle", "stepped", "helicoid"], f"wrong trajectoty -> {trajectory_type}"
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

    crippled_degree = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    state_space = ['rotation_matrix', 'euler', 'position', 'linear_vel', 'angular_vel']
    save_paths = f"./data/rolls{roll_id}"
    config = {
        "max_path_length": 10000,
        "nrollouts": 20,
        'args_init_distribution': {
            'max_radius': 4.2,
            'max_ang_speed': 30,
            'max_radius_init': 0.0,
            'angle_rad_std': 0.6,
            'angular_speed_mean': 0.0,
            'angular_speed_std': 2.0,
        },
        "environment": {
            "state_space_names": state_space,
        },
        "trajectory_args"   :   {
            "wave"          :   trajectory_type,
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

        #assert len(configsfiles) ==0, 'The folder is busy, select other'
        if len(files_paths) > 0:
            user_input = input(f"fold <rolls{roll_id}> busy, want to use roll-id -> <rolls{roll_id + 1}>? yes/no >\t")
            if user_input.lower() == "yes":
                roll_id += 1
                save_paths = f"./data/rolls{roll_id}"
                logger = Logger(save_paths, 'logtest.txt')
            else:
                raise FileExistsError(f"Folder <{save_paths}> already exists")
        #assert len(files_paths)==0, f"The folder <{save_paths}> is busy, select another one"
        if not os.path.exists(save_paths):
            os.makedirs(save_paths)
        
        with open(os.path.join(save_paths, 'experiment_config.json'), 'w') as fp:
            json.dump(config, fp, indent=2)

    print("="* 20 + "ARGS" + "="*20)
    print("save_folder\t->\t", save_paths)
    print("trajectory_type ->\t", trajectory_type)
    print("nrollsouts\t->\t", config["nrollouts"])
    print("early_stop\t->\t", not not_early_stop)
    print("="*50)

    rollouts(
        env, 
        controller, 
        nrolls, 
        max_path_length, 
        save_paths, 
        trajectory,
        logger=logger,
        inject_failure=inject_failure,
        run_all_steps=not_early_stop,
    )
