import os
from typing import List
import joblib
import numpy as np
from ftc.utils.gen_trajectories import Trajectory
from ftc.indi.parameters import Parameters
from ftc.indi.controller import INDIController
from wrapper import QuadrotorEnvRos, state_space 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parser args')
    parser.add_argument('--debug', '-d', type=bool, default=False, dest='debug')

    args = parser.parse_args()
    debug = args.debug

    target_pos = np.array([0, 0, 6.0], dtype=np.float32)
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

    controller = INDIController(parameters=parameters, T_sampling=Ts, state_space=env.state_space)

    max_path_length = 10000

    obs = env.reset()
    cum_reward = 0
    if debug:
        import pdb; pdb.set_trace()

    trajectory_manager = Trajectory(max_path_length)
    trajectory = trajectory_manager.gen_points('point', z_bias=3, nrounds=2)
    #controller.init_controller(state, inputs, 0)
    controller.init_controller(env.last_observation, trajectory[0], parameters.fail_id, 0)
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    for i in range(max_path_length):
        #control_signal = controller(state, inputs)
        control_signal = controller.get_action(obs, trajectory[i], env.last_observation)


        #control_signal[3] = min(control_signal[3], 100)
        #print("[{:.1f}, {:.1f}, {:.1f}, {:.1f}]".format(*list(control_signal)))
        #print("signal rotor 4 --> {:.1f}".format(control_signal[-1]))
        obs, reward, done, info  = env.step(control_signal, trajectory[i])#pos_invert_yz(trajectory[i]))
        if done:
            env.reset()
            break
        observations.append(obs)
        actions.append(control_signal)
        #state.update(env.last_observation)
        cum_reward += reward

    paths = {
        'observations': np.vstack(observations),
        'actions': np.vstack(actions),
        'trajectory': trajectory,
        'info': parameters
    }
    save_paths = './data'
    joblib.dump(paths, os.path.join(save_paths, 'paths9.pkl'))
    print('Cumulative reward {} in {} steps'.format(cum_reward, i))
    print('Normalized Reward: {}'.format(cum_reward/(4 * max_path_length)))