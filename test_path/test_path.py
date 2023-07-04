import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mbrl.utils.process_config import ProcessConfig
from plot_utils.analize_paths import (
    plot_forces, sanity_check_path, plot_3Dtrajectory, 
    plot_trajectory, plot_pos_over_time, plot_ang_velocity_otime,
    plot_euler_over_time
)
from plot_utils.analize_paths import (
    plot_pos_over_time_axes, plot_ang_velocity_otime_axes,
    plot_euler_over_time_axes, plot_grid_3dtrajectories,
    plot_col_summary_trajectory, plot_lin_velocity_otime_axes,
    plot_ndes_over_time_axes
)
from data_analisys import PathsExtractor 
from plot_utils.analisys_distributions import GeneratePlotsDistributions
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roll-id", dest="roll_id", type=int, default=61)
    parser.add_argument("--path-list", "-pl", dest="path_list", nargs="*", type=int, default=1)

    args = parser.parse_args()
    global_path =   './data/'
    roll_idx    =   args.roll_id
    paths_list  =   [args.path_list] if isinstance(args.path_list, int) else args.path_list

    experiment_path = os.path.join(global_path, f"rolls{roll_idx}")
    with open(os.path.join(experiment_path, 'experiment_config.json')) as fp:
        config_exp  =   json.load(fp)
        config      =   config_exp

    names   =   config['environment']['state_space_names']
    #mask    =   [1,1,1,1]#config['crippled_rotor']['mask_quadrotor_1']
    mask    =   np.ones(4)#config_exp['crippled_rotor']
    max_path_length =   config_exp['max_path_length']
    #sanity_check_path(train_path, roll_idx,  paths_list[0])
    import pdb;pdb.set_trace()
    path_extractor   =   PathsExtractor(global_path, f"rolls{roll_idx}", paths_list)
    # Data used to plot
    actions         =   path_extractor.get_actions(concat=False)
    positions       =   path_extractor.get_abs_position(concat=False)
    targets         =   path_extractor.get_targets(concat=False)
    angular_speed   =   path_extractor.get_angular_velocities(concat=False)
    euler           =   path_extractor.get_euler(concat=False)
    linear_speed    =   path_extractor.get_linear_velocities(concat=False)
    try:
        ndes            =   path_extractor.get_ndes(concat=False)
        yaw_reference   =   path_extractor.get_yaw_speed_cmd(concat=False)
    except:
        ndes            = None
        print('NDes not detected!')

    # Using the plot functions
    plot_forces(actions, mask, max_path_length)
    #plot_trajectory(positions, targets)
    #plot_pos_over_time(positions, targets, max_path_length)
    #plot_ang_velocity_otime(angular_speed, max_path_length)
    #plot_euler_over_time(euler, max_path_length)
    #plot_3Dtrajectory(positions, targets, ['g','b'])
    #plt.show()
    fig, axes = plt.subplots(4 if ndes is None else 5, 3, figsize=(12,9))
    plot_pos_over_time_axes(positions, targets, axes[0], max_path_length)
    plot_ang_velocity_otime_axes(angular_speed, axes[1], max_path_length)
    plot_euler_over_time_axes(euler, axes[2], max_path_length)
    plot_lin_velocity_otime_axes(linear_speed, axes[3], max_path_length)
    if ndes is not None:
        [plt.plot(yaw_ref) for yaw_ref in yaw_reference]
    #plot_ndes_over_time_axes(ndes, axes[4], max_path_length)

    plt.tight_layout()
    plt.show()

    #plot_col_summary_trajectory(positions, targets, angular_speed, euler, (4, 3, 1))

    def extraction(train_path, roll_id, paths_list):
        pc      =   ProcessConfig(train_path+'rolls' + str(roll_id) + '/experiment_config.json')
        mask    =   np.array(pc.mask, dtype=np.float32)
        path_extractor   =   PathsExtractor(train_path, 'rolls' + str(roll_id), paths_list)
        # Data used to plot
        actions         =   path_extractor.get_actions(concat=False)
        actions         =   [acts * mask for acts in actions]
        positions       =   path_extractor.get_abs_position(concat=False)
        targets         =   path_extractor.get_targets(concat=False)
        angular_speed   =   path_extractor.get_angular_velocities(concat=False)
        euler           =   path_extractor.get_euler(concat=False)

        return {
            'actions': actions,
            'positions': positions, 
            'targets': targets,
            'angular_speed': angular_speed,
            'euler': euler
        }

    def create_data_list(train_path_list, roll_id_list, paths_lists):
        response    =   []
        for compress in zip(train_path_list, roll_id_list, paths_lists):
            response.append(extraction(*compress))
        
        positions_curves, targets_curves, a_speed_curves, euler_curves, actions_list = [], [], [], [], []

        for dict_response in response:
            positions_curves.append(dict_response['positions'])
            targets_curves.append(dict_response['targets'])
            a_speed_curves.append(dict_response['angular_speed'])
            euler_curves.append(dict_response['euler'])
            actions_list.append(dict_response['actions'])
        
        return positions_curves, targets_curves, a_speed_curves, euler_curves, actions_list


    train_path_list =   3 * ['./data/']
    roll_id_list    =   [28, 22, 25]
    paths_list      =   [[18], [4], [9]]
    titles          =   ['Sin-vertical', 'Circle', 'Helicoid']

    plot_grid_3dtrajectories(*create_data_list(train_path_list, roll_id_list, paths_list), titles=titles)
    plt.tight_layout()
    plt.show()
