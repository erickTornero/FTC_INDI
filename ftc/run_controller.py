import numpy as np
from wrapper import QuadrotorEnvRos, state_space 

target_pos = np.array([0, 0, 6.0], dtype=np.float32)
crippled_degree = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)
state_space = ['rotation_matrix', 'euler', 'position', 'linear_vel', 'angular_vel']

args_init_distribution = {
        'max_radius': 8.2,
        'max_ang_speed': 30,
        'max_radius_init': 6,
        'angle_rad_std': 0.6,
        'angular_speed_mean': 0,
        'angular_speed_std': 1.0,
}

env = QuadrotorEnvRos(np.zeros(3, dtype=np.float32), crippled_degree, state_space, 100, **args_init_distribution)

controller = None#Controller()

max_path_length = 1000

obs = env.reset()
cum_reward = 0
for i in range(max_path_length):
    control_signal = controller(obs)
    obs, reward, done, info,  = env.step(control_signal)
    cum_reward += reward
