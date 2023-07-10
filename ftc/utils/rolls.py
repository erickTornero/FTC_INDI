from collections import deque
from typing import Optional
import numpy as np
import joblib
import os
import glob
#from mbrl.utils.ornstein import OrnsteinUhlenbeck
from ftc.indi import INDIController
from ftc.utils.logger import Logger
from ftc.utils.state import State
from ftc.utils.inputs import Inputs
from ftc.utils.transforms import pos_invert_yz
from wrapper.wrapper_crippled import QuadrotorEnvRos
from ftc.ffree import HoverController
def rollouts(
    env: QuadrotorEnvRos,
    controller: INDIController,
    n_rolls=20,
    max_path_length=5000,
    save_paths=None,
    traj=None,
    initial_states=dict(pos=None,ang=None),
    run_all_steps=False,
    logger: Optional[Logger]=None,
    inject_failure=None
):
    """ Generate rollouts for testing & Save paths if it is necessary"""
    paths           =   []
    allow_inject    =   inject_failure['allow']
    push_failure_at =   None
    damaged_motor   =   -1 # -1 for non damaged
    damaged_motor   =   inject_failure['damaged_motor_index']
    print(f"damaged motor: {damaged_motor}")
    if allow_inject:
        damaged_motor = inject_failure['damaged_motor_index']
        ffree_controller    =   HoverController(env.state_space)
        if inject_failure['type']=='ornstein':
            orstein         =  1#OrnsteinUhlenbeck(**inject_failure['ornstein_uhlenbeck'])
            noise           =  orstein.get_ornstein_noise_length(max_path_length)
            push_failure_at = None
            print('Ornstein-Unlenbeck process failure')
        elif inject_failure['type']=='push':
            push_failure_at = inject_failure['push_failure_at']
            push_new_task   =   inject_failure['push_new_task']

    if save_paths is not None:
        pkls    =   glob.glob(os.path.join(save_paths, '*.pkl'))
        assert len(pkls) == 0, "Selected directory is busy, please select other"
        log_path    =   os.path.join(save_paths, 'log.txt')
        texto   =   'Prepare for save paths in "{}"\n'.format(save_paths)

        print('Prepare for save paths in "{}"'.format(save_paths))
    
    logger.log('Initial states are Fixed!' if initial_states['pos'] is not None else 'Initial states are selected Randomly')
    logger.log('Allowed to execute all t-steps' if run_all_steps else 'Early stop activated')
    #env.set_targetpos(np.random.uniform(-1.0, 1.0, size=(3,)))
    cum_rewards =   []
    total_timesteps =   []
    for i_roll in range(1, n_rolls+1):
        #targetposition  =   np.random.uniform(-1.0, 1.0, size=(3))
        #if traj is None:
        #    targetposition  =   0.8 * np.ones(3, dtype=np.float32)
        #else:
        targetposition  =   traj[0]
        
        next_target_pos =   targetposition
        init_pos    =   initial_states['pos']
        init_ang    =   initial_states['ang']
        #obs = env.reset(init_pos, init_ang)
        #TODO: Hardcoded get task to just the current at index 1 or motor2
        if push_failure_at:
            print('Init without failures')
            env.set_task(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
            env.set_reward_function('type1')
        quadrotor_target = targetposition.copy()
        #quadrotor_target[1:] = -quadrotor_target[1:]
        env.set_targetpos(quadrotor_target)
        obs     = env.reset()
        #task    =   env.get_current_task()[1]
        #mpc.restart_mpc(crippled_factor=task)
        done = False
        timestep    =   0
        cum_reward  =   0.0

        running_paths=dict(observations=[], actions=[], rewards=[], dones=[], next_obs=[], target=[])

        #state = State(invert_axis=True)
        state = State(invert_axis=False)
        state.update(env.last_observation)
        if damaged_motor >= 0: state.update_fail_id(damaged_motor)
        inputs = Inputs()
        inputs.update_position_target(traj[timestep])
        inputs.update_yaw_target(0)
        controller.init_controller(state, inputs, 0)

        while not done and timestep < max_path_length:
            
            #if timestep == 120 and traj is None:
            #    next_target_pos  = np.zeros(3, dtype=np.float32)
            #elif traj is not None:            
            if allow_inject:
                if inject_failure['type']=='ornstein':
                    env.set_task(np.array([1.0, noise[timestep], 1.0, 1.0], dtype=np.float32))
                elif inject_failure['type']=='push':
                    if timestep == push_failure_at:
                        #TODO Hardcoded failure
                        print('Insert failure')
                        env.set_task(np.array([1.0, push_new_task, 1.0, 1.0], dtype=np.float32))
                    if timestep == (push_failure_at + 4):
                        print('Switch architecture')
                        # TODO: Switch controller

            next_target_pos =   traj[timestep + 1]
            inputs.update_position_target(next_target_pos)

            #action = mpc.get_action_PDDM(stack_as, 0.6, 5)
            #action = controller(state, inputs)
            action = controller.get_action(obs, next_target_pos, env.last_observation)
            #TODO: Fix order of control signal
            #tmp = action[3]
            #action[3] = action[1]
            #action[1] = tmp

            next_obs, reward, done, env_info =   env.step(action, next_target_pos)#pos_invert_yz(next_target_pos)) TODO, checking without inverting next

            if run_all_steps: done=False

            #if save_paths is not None:
            observation = obs
            running_paths['observations'].append(observation.flatten())
            running_paths['actions'].append(action.flatten())
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
            state.update(env.last_observation)
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
