import os
import joblib
import numpy as np
from ftc.utils.state import State
from ftc.utils.inputs import Inputs
from ftc.utils.gen_trajectories import Trajectory
from ftc.indi.parameters import Parameters
from ftc.indi.controller import INDIController
from wrapper import QuadrotorEnvRos, state_space 

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

controller = INDIController(parameters=parameters, T_sampling=Ts)

max_path_length = 10000
state = State(invert_axis=True)
state.update_fail_id(parameters.fail_id)
inputs = Inputs()
inputs.updatePositionTarget([0, 0, -2])
inputs.update_yawTarget(0)

obs = env.reset()
state.update(env.last_observation)
cum_reward = 0
import pdb; pdb.set_trace()

trajectory_manager = Trajectory(max_path_length, -3)
trajectory = trajectory_manager.gen_points('helicoid', 2)
#for _ in range(2):
#    obs, _, _, _, = env.step(np.array([400, 0, 400, 0]))
#    state.update(env.last_observation)
controller.init_controller(state, inputs, 0)
observations = []
actions = []
for i in range(max_path_length):
    inputs.updatePositionTarget(trajectory[i])
    ##if i == 5000:
    ##    print('setposition')
    ##    import pdb; pdb.set_trace()
    ##    inputs.updatePositionTarget([0, 1, -2])
    control_signal = controller(state, inputs)
    #print("[{:.1f}, {:.1f}, {:.1f}, {:.1f}]".format(*list(control_signal)))
    #print("signal rotor 4 --> {:.1f}".format(control_signal[-1]))
    tmp = control_signal[3]
    control_signal[3] = control_signal[1]
    control_signal[1] = tmp

    #control_signal[3] = min(control_signal[3], 100)
    print("[{:.1f}, {:.1f}, {:.1f}, {:.1f}]".format(*list(control_signal)))
    #print("signal rotor 4 --> {:.1f}".format(control_signal[-1]))
    obs, reward, done, info  = env.step(control_signal)
    if done:
        env.reset()
        break
    observations.append(obs)
    actions.append(control_signal)
    state.update(env.last_observation)
    cum_reward += reward

paths = {
    'observations': np.vstack(observations),
    'actions': np.vstack(actions),
    'trajectory': trajectory,
    'info': parameters
}
save_paths = './data'
joblib.dump(paths, os.path.join(save_paths, 'paths9.pkl'))