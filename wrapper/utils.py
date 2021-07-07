from __future__ import division

import warnings
import math

import numpy

_EPS = numpy.finfo(float).eps * 4.0

_NEXT_AXIS = [1, 2, 0, 1]

_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    Note that many Euler angle triplets can describe one matrix.
    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True
    """
    q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)


def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.
    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True
    """
    rotmat=None
    if quaternion.shape[0]==9: 
        rotmat=quaternion.reshape(3,3)
        assert rotmat.shape == (3, 3), "Wrong dimension rotation_matrix"

    return  euler_from_matrix(quaternion_matrix(quaternion) if rotmat is None else rotmat)
    #return euler_from_matrix(quaternion_matrix(quaternion), axes)


def quaternion_matrix_batch(quaternion):
    batch_size  =   quaternion.shape[0]
    q = numpy.array(quaternion[:, :4], dtype=numpy.float64, copy=True)
    nq = numpy.sum(q*q, axis=1).reshape(-1, 1)
    factor  =   1.0 * (nq < _EPS)
    nq  +=  factor * 0.1
    q *= numpy.sqrt(2.0 / nq)
    # Batch outer sumation using Einsten Summation conventions
    q   =   numpy.einsum('ab,ad->abd', q, q)
    
    L   =   [
        1.0-q[:, 1, 1] - q[:, 2, 2],     q[:, 0, 1]-q[:, 2, 3],     q[:, 0, 2] + q[:, 1, 3],
            q[:, 0, 1] + q[:, 2, 3], 1.0-q[:, 0, 0]-q[:, 2, 2],     q[:, 1, 2] - q[:, 0, 3],
            q[:, 0, 2] - q[:, 1, 3],     q[:, 1, 2]+q[:, 0, 3], 1.0-q[:, 0, 0] - q[:, 1, 1]
    ]
    L = [numpy.expand_dims(col, axis=1) for col in L]
    identity_flatten    =   numpy.repeat(numpy.expand_dims(numpy.identity(3).flatten(), axis=0), batch_size, axis=0)
    quat1   =   numpy.concatenate(L, axis=1)

    return (1.0 - factor) * quat1 + factor * identity_flatten


def euler_from_quaternion_batch(quaternion, axes='sxyz'):
    return euler_from_matrix_batch(quaternion_matrix_batch(quaternion))





def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = numpy.empty((4, ), dtype=numpy.float64)
    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss
    if parity:
        quaternion[j] *= -1

    return quaternion

def euler2quat(roll, pitch, yaw):

    qx = numpy.sin(roll/2) * numpy.cos(pitch/2) * numpy.cos(yaw/2) - numpy.cos(roll/2) * numpy.sin(pitch/2) * numpy.sin(yaw/2)
    qy = numpy.cos(roll/2) * numpy.sin(pitch/2) * numpy.cos(yaw/2) + numpy.sin(roll/2) * numpy.cos(pitch/2) * numpy.sin(yaw/2)
    qz = numpy.cos(roll/2) * numpy.cos(pitch/2) * numpy.sin(yaw/2) - numpy.sin(roll/2) * numpy.sin(pitch/2) * numpy.cos(yaw/2)
    qw = numpy.cos(roll/2) * numpy.cos(pitch/2) * numpy.cos(yaw/2) + numpy.sin(roll/2) * numpy.sin(pitch/2) * numpy.sin(yaw/2)

    return [qx, qy, qz, qw]


def euler_from_matrix_batch(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    Note that many Euler angle triplets can describe one matrix.
    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    if repetition:
        sy = numpy.sqrt(M[:, 3*i + j]*M[:, 3 * i + j] + M[:, 3 * i + k]*M[:, 3 * i+ k])
        factor  =   1.0 * (sy  > _EPS)
        numerator   =   factor * M[:, 3 * i + j] - (1.0 - factor) * M[:, 3 * j + k]
        denominator =   factor * M[:, 3 * i + k] + (1.0 - factor) * M[:, 3 * j + j]

        ax      =   numpy.arctan2(numerator, denominator)
        ay      =   numpy.arctan2(sy, M[:, 3 * i + i])
        az      =   factor * numpy.arctan2( M[:, 3 * j + i], -M[:, 3 * k + i])

    else:
        cy = numpy.sqrt(M[:, 3 * i + i]*M[:, 3 * i + i] + M[:, 3 * j + i]*M[:, 3 * j + i])
        factor  =   1.0 * (cy > _EPS)
        numerator   =   factor * M[:, 3 * k + j] - (1 - factor) * M[:, 3 * j + k]
        denominator =   factor * M[:, 3 * k + k] + (1 - factor) * M[:, 3 * j + j]
        ax      =   numpy.arctan2(numerator, denominator)
        ay      =   numpy.arctan2(-M[:, 3 * k + i], cy)
        az      =   factor * numpy.arctan2( M[:, 3 * j + i],  M[:, 3 * i + i])

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    
    ax, ay, az  =   ax.reshape(-1, 1), ay.reshape(-1, 1), az.reshape(-1, 1)
    return numpy.concatenate((ax, ay, az), axis=1)



import rospy
import xml.etree.ElementTree as ET
class RobotDescription:
    
    def __init__(self, name):
        self.name   =   name
        print(self.name)
    def _get_max_vel(self):
        try:
            params  =   rospy.get_param('/robot_description') 
        except Exception as e:
            print(e)
            params  =   rospy.get_param('/'+ self.name+'/robot_description')

        tree    =   ET.XML(params)
        gazebo_plugins      =   [gz.find('plugin') for gz in tree.findall('gazebo')]
        max_rotor_speed     =   None  
        # TODO: HardCodded!!!
        rotor_name = self.name + '_front_motor_model'
        for plugin in gazebo_plugins:
            if plugin is not None:
                name_plug = plugin.attrib['name']
                if name_plug == rotor_name:
                    max_rotor_speed = float(plugin.find('maxRotVelocity').text)
                    return max_rotor_speed



