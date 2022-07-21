import os
from time import time
from ftc.switch_controllers.switch_controller import SwitchController, MODE_CONTROLLER
from ftc.utils.gen_trajectories import Trajectory
from ftc.utils.logger import Logger
from wrapper.wrapper_crippled import QuadrotorEnvRos


# Define rolls function to inject failures at specific timestep
import joblib
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
        "nrounds"       :   2,
        "z_bias"        :   -6.0,
    },
    "controllers": {
        'fault_free': 'HoverController',
        'fault_case': 'indi',
    },
    "inject_failure"    :   {
        "allow": False,
        "type": "push",  # "push" or "ornstein"
        "damaged_motor_index": 2, #allowed [0, 1, 2, 3]
        "push_failure_at": 5000,
        "push_new_task": 0.0,
        "ornstein_uhlenbeck": {
            "delta": 0.03,
            "sigma": 0.20, 
            "ou_a":  0.5,
            "ou_mu": 0.78, 
            "seed":  42
        }
    },
    'state_space_names': [
        'rotation_matrix', 'position', 'euler', 'linear_vel', 'angular_vel'
    ],
    "switch_info": [
        {
            ""
        }
    ]
}

def exec_rollouts(
    env: QuadrotorEnvRos,
    trajectory_info: dict,
    controllers_info: dict,
    failure_info: dict,
    nrolls: int=20,
    max_path_length: int=5000,
    save_paths: os.path=None,
    logger: Optional[Logger]=None,
    run_all_steps: bool=False,
):  
    #Failure variables initialization
    DELAY_TIMESTEPS =   4
    allow_inject    =   failure_info['allow']
    type_failure    =   failure_info['type']
    if allow_inject:
        push_failure_at =   failure_info['push_failure_at'] if type_failure == 'push' else -1e5
        #noise_ornstein  = None
        #raise NotImplementedError('Not implemented ornstein noise') if type_failure else -1
    else:
        push_failure_at = -1e5
        noise_ornstein = None
    
    # create trajectory
    trajectory = Trajectory(max_path_length).gen_points(**trajectory_info)

    controller = SwitchController(env, MODE_CONTROLLER.FAULT_FREE)

    paths           =   []
    cum_rewards     =   []
    total_timesteps =   []
    
    for i_roll in range(1, nrolls + 1):
        env.set_task(np.ones(4))
        targetposition = trajectory[0]
        running_paths=dict(observations=[], actions=[], rewards=[], dones=[], next_obs=[], target=[])

        obs         =   env.reset()
        done        =   False
        timestep    =   0
        cum_reward  =   0
        while not done and timestep < max_path_length:
            if (push_failure_at == timestep): env.set_task(np.array([1.0, 1.0, 0.0, 1.0])); print('setting env')
            if (push_failure_at + DELAY_TIMESTEPS) == timestep:
                controller.switch_faulted(targetposition, damaged_rotor_index=2, c_time=0)
            
            next_target_pos =   trajectory[timestep + 1]
            control_signal = controller.get_action(obs, next_target_pos)
            next_obs, reward, done, env_info = env.step(control_signal, next_target_pos)
            if run_all_steps: done=False
            #if save_paths is not None:
            observation = obs
            running_paths['observations'].append(observation.flatten())
            running_paths['actions'].append(control_signal.flatten())
            running_paths['rewards'].append(reward)
            running_paths['dones'].append(done)
            running_paths['next_obs'].append(next_obs)
            running_paths['target'].append(targetposition)

            if done or len(running_paths['rewards']) >= max_path_length:
                paths.append(dict(
                    observation=np.asarray(running_paths['observations']),
                    actions=np.asarray(running_paths['actions']),
                    rewards=np.asarray(running_paths['rewards']),
                    dones=np.asarray(running_paths['dones']),
                    next_obs=np.asarray(running_paths['next_obs']),
                    target=np.asarray(running_paths['target'])
                ))
            # endif
            
            targetposition  =   next_target_pos
            obs = next_obs
            # Test stacked
            cum_reward  +=  reward
            timestep += 1
        
        logger.log('{} rollout, reward-> {} in {} timesteps'.format(i_roll, cum_reward, timestep))
        if save_paths is not None:
            joblib.dump(paths, os.path.join(save_paths, 'paths.pkl'))
            logger.save()
        total_timesteps.append(timestep)
        cum_rewards.append(cum_reward)
    
    npaths  =   len(cum_rewards)
    logger.log('Mean reaward over {} paths -->\t{}, mean timesteps-> {}'.format(npaths, sum(cum_rewards)/npaths, sum(total_timesteps)/npaths))
    logger.save()
    return paths