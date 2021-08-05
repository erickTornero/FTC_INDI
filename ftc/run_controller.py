import numpy as np
from ftc.state import State
from ftc.inputs import Inputs
from ftc.parameters import Parameters
from ftc.controller import INDIController
from wrapper import QuadrotorEnvRos, state_space 

target_pos = np.array([0, 0, 6.0], dtype=np.float32)
crippled_degree = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
state_space = ['rotation_matrix', 'euler', 'position', 'linear_vel', 'angular_vel']

args_init_distribution = {
        'max_radius': 10,
        'max_ang_speed': 30,
        'max_radius_init': 0,
        'angle_rad_std': 0.1,
        'angular_speed_mean': 0,
        'angular_speed_std': 0.1,
}

quadrotor_parms_path = 'params/quad_parameters.json'
control_parms_path = 'params/control_parameters.json'
parameters = Parameters(quadrotor_parms_path, control_parms_path)
rate = parameters.freq
Ts = 1/rate
crippled_degree[parameters.fail_id] = 0.0

env = QuadrotorEnvRos(np.zeros(3, dtype=np.float32), crippled_degree, state_space, rate, **args_init_distribution)

controller = INDIController(parameters=parameters, T_sampling=Ts)

max_path_length = 5000
state = State()
state.update_fail_id(parameters.fail_id)
inputs = Inputs()
inputs.updatePositionTarget([1, 1, -2])
inputs.update_yawTarget(0)

obs = env.reset()
state.update(env.last_observation)
cum_reward = 0
import pdb; pdb.set_trace()
for i in range(max_path_length):
    control_signal = controller(state, inputs)
    tmp = control_signal[3]
    control_signal[3] = control_signal[1]
    control_signal[1] = tmp

    obs, reward, done, info,  = env.step(control_signal)
    if done:
        env.reset()
        break
    state.update(env.last_observation)
    cum_reward += reward
