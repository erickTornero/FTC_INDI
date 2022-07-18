import numpy as np
from wrapper import QuadrotorEnvRos
from ftc.ffree import HoverController#, HoverLQRff
#def rollouts(env, controller, frequency, nrolls, max_path_length, save_paths, trajectory, initial_states, run_all_steps=False, logger=None, inject_failure=None):
def get_from_vector(vector, start, end):
    return vector[start:end]


def get_values(vector, indexes, state_space_names):
    return (get_from_vector(vector, **indexes[name]) for name in state_space_names)

def get_actions_from_control(obs):
    return np.random.uniform(0, 838, 4)

target_pos  =   np.array([0, 0, 2], dtype=np.float32)
crippled_degree = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

state_space_names =   ['rotation_matrix', 'position', 'euler', 'linear_vel', 'angular_vel']
frequency   =   20
initial_conditions_params = {
    "max_radius": 8.2,
    "max_ang_speed": 30.0,
    "max_radius_init": 0.0,
    "angle_rad_std": 0.0,
    "angular_speed_mean": 0.0,
    "angular_speed_std": 0.0
}
flight_time =   12.5  # in seconds
max_path_length =   int(flight_time * frequency)
env =   QuadrotorEnvRos(target_pos, crippled_degree, state_space_names, frequency, **initial_conditions_params)
hover_control   =   HoverController()#HoverLQRff()


def roll(env:QuadrotorEnvRos, controller: HoverController, max_path_length: int):    
    indexes =   env.state_space.get_state_space_indexes()
    obs     =   env.reset()
    cumulative_reward   =   0
    #env.target_pos      =   np.array([0, 1, 2], dtype=np.float32)

    for i in range(max_path_length):
        if i==0:
            continue
        action  =   controller.compute_rotors_speeds_v2(*get_values(obs, indexes, state_space_names), target_pos)
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            return cumulative_reward, i
    return cumulative_reward, max_path_length

if __name__ == "__main__":
    cr, i = roll(env, hover_control, max_path_length)
    print('Cumulative reward {} in {} steps'.format(cr, i))
    print('Normalized Reward: {}'.format(cr/(4 * max_path_length)))