import numpy as np
from ftc.utils.state import State
from ftc.utils.inputs import Inputs
from ftc.lqr.controller import LQRController
from ftc.lqr.parameters import Parameters
from wrapper import QuadrotorEnvRos, state_space

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
control_parms_path = 'params/control_params_lqr.json'
parameters = Parameters(quadrotor_parms_path, control_parms_path)
rate = parameters.freq
Ts = 1/rate

# 
if parameters.fail_id>=0:
    crippled_degree[parameters.fail_id] = 0.0

env = QuadrotorEnvRos(np.zeros(3, dtype=np.float32), crippled_degree, state_space, rate, **args_init_distribution)

controller = LQRController(parameters=parameters, T_sampling=Ts)

max_path_length = 50000
inputs = Inputs()
inputs.update_position_target(np.array([0, 0 , 5]))
inputs.update_yawTarget()
state = State()
state.update_fail_id(parameters.fail_id)

obs = env.reset()
cum_reward = 0

state.update(env.last_observation)
for i in range(max_path_length):
    control_signal = controller(state, inputs)
    
    obs, reward, done, info  = env.step(control_signal) # make an step in the environment
    state.update(env.last_observation) #map states to State class
    cum_reward += reward
    if done:
        env.reset()
        break


"""
    state = State() to get the current state at each time-step.
    check the file ftc/utils/state.py to get a notion how to use it

    inputs = Inputs(), use it to pass targets to the control algorithm
    check the file ftc/utils/inputs.py to get a notion how to use it

    files you probably will need to modify
    - params/control_params_lqr.json, put here all the parameters of your LQR controller
    - ftc/lqr/parameters.py, map here the values of the previous json to ease coding
    - ftc/lqr/controller.py, put here the core of your LQR, the object must return the control signal in a forward pass, check __call__ function
"""