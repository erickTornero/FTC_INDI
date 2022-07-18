
import numpy as np
from collections import OrderedDict
from .base_wrapper_ros import BaseWrapperROS
from gym import spaces
from mav_msgs.msg import Actuators
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from .utils import euler_from_quaternion, quaternion_matrix, RobotDescription, quaternion_from_euler
from .state_space import StateSpaceRobots
import rospy
#from tf.transformations import quaternion_from_euler, euler_from_quaternion
from random import gauss

class WrapperROSQuad(BaseWrapperROS):
    """
    Wrapper for quadrotor
    To change the behaiviour edit 
    @state_space_name_spaces
    @action_space shape

    @args: 
    -target_pos: target position the position in the observation space is relative postion to the target
    -rate: The rate the controller works on.
    """

    """
        You can define the <state space> with the following names
        ['rotation_matrix', 'position', 'euler', 'quaternion' 
            'linear_vel', 'angular_vel]
    """

    def __init__(self, model_name='hummingbird', target_pos=np.zeros(3, np.float32), rate=100, state_space_names=['rotation_matrix', 'position', 'linear_vel', 'angular_vel'], **kwargs_init):
        
        
        super(WrapperROSQuad, self).__init__(rate)
        self.model_name         =   model_name
        self.state_space        =   StateSpaceRobots(state_space_names)
        shape_state_space       =   self.state_space.get_state_space_shape()
        robot_description       =   RobotDescription(self.model_name)
        max_velocity_rotor      =   robot_description._get_max_vel()
        rospy.loginfo('Found Rotor max speed {}'.format(max_velocity_rotor))
        rospy.loginfo('State space shape: {} with following meassurements: \n{}'.format(shape_state_space, '-\t'+'\n-\t'.join(self.state_space.names)))
        self.last_observation   =   None
        self.action_space       =   spaces.Box(low=0.0, high=max_velocity_rotor, shape=(4,), dtype=np.float32)
        self.observation_space  =   spaces.Box(low=-np.inf, high=np.inf, shape=shape_state_space, dtype=np.float32)
        self.target_pos         =   target_pos
        #print(self.target_pos)

        self.actuator_msg       =   Actuators()        
        
        self._sensors_topic_name    =   '/hummingbird/ground_truth/odometry'
        self._actuator_reader     =   '/hummingbird/motor_speed'
        self._imu_topic_name    =   '/hummingbird/imu'
        #TODO: Fill with correspondent ROS topics
        self._actuators_pub     =   rospy.Publisher('hummingbird/command/motor_speed', Actuators, queue_size=10)
        self._set_states_srv    =   rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        #self._states_pub        =   None
        self._sensors_subs      =   rospy.Subscriber('/hummingbird/ground_truth/odometry', Odometry, self._callback_sensor_meassurements)
        self._imu_subs          =   rospy.Subscriber(self._imu_topic_name, Imu, self._callback_imu_meassurements)

        self.max_radius         =   kwargs_init['max_radius']
        self.max_ang_speed      =   kwargs_init['max_ang_speed']
        self.max_radius_init    =   kwargs_init['max_radius_init']
        self.angle_std          =   kwargs_init['angle_rad_std']
        self.angular_speed_mean =   kwargs_init['angular_speed_mean']
        self.angular_speed_std  =   kwargs_init['angular_speed_std']
        self.counter = 0
        self._imu = None
        #self._check_all_systems_ready()

    def _set_action_msg(self, action, actuator_publisher=None):
        self.actuator_msg.angular_velocities    =   action
        self._actuators_pub.publish(self.actuator_msg) if actuator_publisher is None else actuator_publisher.publish(self.actuator_msg)

    def _call_setstates_srv(self, position=None, orientation=None, angular_speed=None, _name=None):
        model_state =   ModelState()
        model_state.model_name  =   self.model_name if _name is None else _name
        if position is not None: model_state.pose.position   =   Point(*position)
        if orientation is not None: model_state.pose.orientation    =   Quaternion(*orientation)
        if angular_speed is not None: model_state.twist.angular = Vector3(*angular_speed)
        
        success = self._set_states_srv.call(model_state).success
        return success
    
    def _callback_sensor_meassurements(self, data):
        """
        Returns sensor meassurements data
        and pass to class variables
        """
        # elements of odometry
        self._odometry      =   data
        self.counter += 1
        if len(self.actuator_msg.angular_velocities) > 0:
            self._actuators_pub.publish(self.actuator_msg)
    
    def _callback_imu_meassurements(self, data):
        """
        Returns sensor meassurements data
        and pass to class variables
        """
        # elements of odometry
        self._imu      =   data

    def _set_states(self, position=None, attitude=None, targetpos=None, name=None):
        if targetpos is None: targetpos = self.target_pos
        if position is None or attitude is None:
            init_pos, init_att  =   self._get_random_pose(max_radius_init=self.max_radius_init, max_angle=self.angle_std, respecto=targetpos)
            init_ang_speed      =   self._get_gauss_angular_speed(self.angular_speed_mean, self.angular_speed_std)
            #print('Sampled_pos      : \t{}'.format(init_pos))
            #print('Sampled attitude : \t{}'.format(init_att))
            #print('Sampled ang speed: \t{}'.format(init_ang_speed))
            if position is None: position=init_pos
            if attitude is None: attitude=init_att
            #TODO: missing argument of the initial angular speed
        success =   self._call_setstates_srv(position, attitude, init_ang_speed, name)
        return success
        # TODO: Validate

    def _compute_done(self, states_dict, targetpos=None):
        position    =   states_dict['position']
        distance    =   (targetpos - position) if targetpos is not None else position
        distance    =   np.sqrt((distance * distance).sum())
        # Add Early stop when angular velocity in x or y >= 10rad/s
        ang_vel     =   states_dict['angular_vel']
        ang_vel_es  =   np.abs(ang_vel[0]) >=self.max_ang_speed or np.abs(ang_vel[1]) >= self.max_ang_speed
        # TODO: Specify well the done 
        # TODO: in this case if dist > 3.2m->EarlyStop

        return distance > self.max_radius or ang_vel_es

    def _compute_rewards(self, states_dict, action, targetpos=None):
        position    =   states_dict['position']
        distance    =   (targetpos - position) if targetpos is not None else (self.target_pos - position)
        distance    =   np.sqrt((distance * distance).sum())

        return 4.0 - 1.25 * distance

    def _before_reset(self):
        """
            We must set the actions to 0, to avoid to get high angular 
            speeds during the reset process. This problem is due to the
            _check_all_systems_ready, takes a considerable amount of time
            then, previously biased actions increase the angular speed
        """
        self._set_action_msg([0.0, 0.0, 0.0, 0.0])
        
    #@staticmethod
    #def _flat_observation_st(obs_dict):
    #    _state_space    =   [obs_dict[name] for name in WrapperROSQuad.state_space_names]
    #    #position    =   obs_dict['position']
    #    #linear_vel  =   obs_dict['linear_vel']
    #    #angular_vel =   obs_dict['angular_vel']
    #    #rot_mat     =   obs_dict['rot_mat']
    #    #[print('shape {}\n'.format(ss.shape)) for ss in _state_space]
    #    return np.concatenate(
    #        _state_space
    #    )
    


    def _flat_observation(self, obs_dict):
        _state_space    =   [obs_dict[name] for name in self.state_space.names]
        
        return np.concatenate(
            _state_space
        )
        #return WrapperROSQuad._flat_observation_st(obs_dict)

    def _get_observation_state(self, _odom=None, targetpos = None):
        """ Get observation states in an ordered dict""" 

        if _odom is None: _odom = self._odometry
        if targetpos is None: targetpos = self.target_pos
        _imu_lecture            =   self._imu
        if _imu_lecture is not None: 
            _lin_acc = _imu_lecture.linear_acceleration
            _lin_acc = np.array([_lin_acc.x, _lin_acc.y, _lin_acc.z], dtype=np.float32)
        else:
            _lin_acc = np.zeros(3, dtype=np.float32)
        #TODO: Get the observation from ROS
        pose                    =   _odom.pose.pose
        #pose covariance        =   data.pose.covariance
        twist                   =   _odom.twist.twist


        _position          =   pose.position
        _quaternion        =   pose.orientation
        # To numpy arrays
        _position          =   np.array([_position.x, _position.y, _position.z], dtype=np.float32)
        _quaternion        =   np.array([_quaternion.x, _quaternion.y, _quaternion.z, _quaternion.w], dtype=np.float32)
        # relative_position
        _position          =   _position# - targetpos

        _linvel            =   twist.linear
        _angvel            =   twist.angular
        # To arrays
        _linvel            =   np.array([_linvel.x, _linvel.y, _linvel.z])
        _angvel            =   np.array([_angvel.x, _angvel.y, _angvel.z])
 
        _rotation_matrix   =   quaternion_matrix(_quaternion)[:3,:3].flatten() if 'rotation_matrix' in set(self.state_space.names) else None
        
        _euler             =   euler_from_quaternion(_quaternion if _rotation_matrix is None else _rotation_matrix) if 'euler' in set(self.state_space.names) else None

        # get wspeeds
        _w_speeds = None
        if self.actuator_msg.angular_velocities is not None:
            _w_speeds = self.actuator_msg.angular_velocities
        observation =   dict(
            position            =   _position,
            quaternion          =   _quaternion,
            linear_vel          =   _linvel,
            angular_vel         =   _angvel,
            rotation_matrix     =   _rotation_matrix,
            euler               =   _euler,
            lin_acc             =   _lin_acc,
            w_speeds            =   _w_speeds,
        )

        self.last_observation   =   observation

        return observation

    def _check_all_systems_ready(self):
        """
        Check if all sensors are ready to work
        """
        _   =   self._check_actuators_ready()
        self._odometry = self._check_all_sensors_ready(self._sensors_topic_name)
        self._imu = self._check_imu_ready()

    def _init_env_variables(self):
        pass


    def _check_all_sensors_ready(self, sensor_name):
        """
            Check if all sensors are available
            for the moment sensor_name is the odometry sensor.
        """
        return self._check_topic_ready(sensor_name, Odometry)
        

    def _check_topic_ready(self, topic_name, message_class):
        """ Check if a topic is available,
            Made for publisher topics
        """
        _topic_data =   None
        while _topic_data is None and not rospy.is_shutdown():
            try:
                _topic_data =   rospy.wait_for_message(topic_name, message_class, timeout=2.0)
                rospy.logdebug('Current topic {} is READY!!'.format(topic_name))
            except rospy.exceptions.ROSException:
                rospy.logerr('Current topic {} is not available'.format(topic_name))
        
        return _topic_data
    
    def _check_imu_ready(self):
        return self._check_topic_ready(self._imu_topic_name, Imu)


    def _check_actuators_ready(self):
        return self._check_topic_ready(self._actuator_reader, Actuators)
        


    def _get_gauss_euler(self, angle_std): 
        x   =   [gauss(0, angle_std) for _ in range(3)]
        x   =   np.clip(x, -np.pi/2.0+0.001, np.pi/2.0-0.001)
        return  np.asarray(x, dtype=np.float32)


    def _get_gauss_angular_speed(self, angular_mean, angular_std):
        v   =   [gauss(angular_mean, angular_std) for _ in range(3)]
        return np.asarray(v, dtype=np.float32)


    def _sample_spherical2cartesian(self, max_radius_init):
        """ 
            Sample position in the spherical notation using uniform distribution.
            Then transform to the cartesian notation
            -max_radius_init:    the 'rho' variable
        """
        rho         =   np.random.uniform(-max_radius_init, max_radius_init)
        theta       =   np.random.uniform(0, np.pi)
        phi         =   np.random.uniform(-np.pi, np.pi)

        x           =   rho * np.sin(theta) * np.cos(phi)
        y           =   rho * np.sin(theta) * np.sin(phi)
        z           =   rho * np.cos(theta)
        return np.array([x, y, z], dtype=np.float32)

    def _get_random_pose(self, max_radius_init=3.2, max_angle=np.pi, respecto:np.ndarray=None):
        """ 
            _get_random_pose: Gets the initial pose: position & attitude
            position: Sample random position using spherical-to-cartesian transformation
            attitude: We sample first a attitude in Euler notation, then we transform to the quaternion notation

            @args:
            -max_radius_init:    Maximum radius taken with respect the the respecto variable (rho).
            -max_angle :    Maximum angle at the initial position (avoid upside samples at the beginning)
            -respecto  :    Position to sample around
        """
        
        if respecto is None:
            respecto    =   np.zeros(3, dtype=np.float32)

        position_sampled    =   self._sample_spherical2cartesian(max_radius_init) + respecto
        attitude_sampled    =   quaternion_from_euler(*self._get_gauss_euler(max_angle))

        return position_sampled, attitude_sampled