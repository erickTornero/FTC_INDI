# Fault Tolerant Controller


A python and ROS, implementation of paper: 
*"Incremental Nonlinear Fault-Tolerant Control of a Quadrotor With Complete Loss of Two Opposing Rotors"*

S. Sun, X. Wang, Q. Chu and C. d. Visser.

# Requirements

- ros melodic
- [rotors_simulator package](https://github.com/ethz-asl/rotors_simulator)
- python3

# Installation
- git clone https://github.com/erickTornero/FTC_INDI.git
- create an environment with python3 >= 3.6
```bash
    pip install -e .
```


## Fault Tolerant Controller using LQR

### Launch ROS Environment

```bash
    roslaunch launchers/launchquad.launch
```

### Run LQR FTC Controller
```bash
    python scripts/run_LQRcontroller.py
```

