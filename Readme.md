# Fault Tolerant Controller


A python and ROS, implementation of paper: 
*"Incremental Nonlinear Fault-Tolerant Control of a Quadrotor With Complete Loss of Two Opposing Rotors"*

S. Sun, X. Wang, Q. Chu and C. d. Visser.

# Requirements

- ros melodic
- python3


# Fault Tolerant Controller using LQR

### Launch ROS Environment

```bash
    roslaunch launchers/launchquad.launch
```

### Run LQR FTC Controller
```bash
    python lqr_ftc/run_controller.py
```

