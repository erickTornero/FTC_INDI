import rospy
import os

def ros_launcher(launch_file):
    cmd =   "roslaunch {}".format(launch_file)
    os.system(cmd)

def lauch_quad(hidden=False, folder_base="ros_launchers"):
    if hidden:
        filename    =   "launch1train.launch"
    else:
        filename    =   "launchquad.launch"

    file_path = os.path.join(folder_base, filename)
    ros_launcher(file_path)

def launch_multiple(folder_base="ros_launchers"):
    file_path   =   os.path.join(folder_base, "launchmultiple.launch")
    ros_launcher(file_path)

def close_ros():
    os.system('killall roscore')