'''
The module is used for the trasnformation from a tilt sphere to a icosahedron face.
pano_orientation: the vector fo tilt
sphere_vec      : the normal vector of the face on icosahedron
'''
from math import tan, acos, cos, sin
import numpy as np


def normalize(vector):
    '''
    Calculate the normalized vector

    Args:
        vector(np.array): the input vector

    Returns:
        vector(np.array): the normalized vector
    '''
    norm = np.sqrt(np.sum(vector ** 2))
    return vector / norm

def sphere2pixel(pano_orientation, sphere_vec, radius):
    '''
    Main trasform function between shere and polyhedron

    Args:
        pano_orientation(np.array)  : the vector for how the ball tilted
        shere_vec(np.array)         : the normal vector of the projection face
        radius(int)                 : the size for the projected sphere

    Returns:
        (x, y) (tuple)              : the coordinate of the panoramic pic
    '''
    pano_orientation = normalize(pano_orientation)
    sphere_vec = normalize(sphere_vec)

    tmp = np.copy(pano_orientation)
    tmp[2] = 0

    # horizon = [-bc, ac, 0]
    horizon = np.cross(pano_orientation, tmp)
    # horizon_norm = [a*c^2, b*c^2, -c*a^2-c*b^2]
    horizon_norm = np.cross(pano_orientation, horizon)
    direction_norm = np.cross(pano_orientation, sphere_vec)
    horizon_norm = normalize(horizon_norm)
    direction_norm = normalize(direction_norm)

    determinant = np.linalg.det(np.vstack((horizon_norm, direction_norm, pano_orientation)))
    theta = np.arctan2(determinant, np.dot(horizon_norm, direction_norm.T))
    radial_angle = acos(np.dot(pano_orientation, sphere_vec.T))
    radial_distance = radius * tan(radial_angle)

    if np.dot(pano_orientation, sphere_vec.T) >= 0:
        return (radial_distance*cos(theta), radial_distance*sin(theta))
    return (-np.inf, -np.inf)
