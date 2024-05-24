import numpy as np
import math


def get_translation_error(gt : np.ndarray, est : np.ndarray, metrics : str = "m") -> float:
    ''' Calculates translation error (euclidean norm).
        Args:
        gt: ground-truth translation vector.
        est: estimated translation vector.
        metrics: avaiable in mm, cm, dm and m.
    Returns:
        Calculated error in given metrics. 
        '''
    gt_copy = np.copy(gt)
    est_copy = np.copy(est)
    met = 1 # default metrics are in meters
    if metrics == "dm":
        met = 10
    elif metrics == "cm":
        met = 100
    elif metrics == "mm":
        met = 1000
    gt_copy *= met
    est_copy *= met
    return np.linalg.norm(gt_copy - est_copy)

def get_rotation_error(gt : np.ndarray, est : np.ndarray, degrees: bool = True) -> float:
    '''Calculates the rotation error.
        Args:
        gt: ground-truth rotation matrix.
        est: estimated rotation matrix.
        degrees: flag whether return result in radians or in degrees.
    Returns:
        Calculated error in radians/degress. 
        '''
    inv_gt = np.linalg.inv(gt)
    m_mult = np.matmul(est, inv_gt)
    m_trace = (np.trace(m_mult) -1.0) / 2.0
    m_trace = min(1.0, max(-1.0, m_trace))
    if degrees:
        return math.degrees(math.acos(m_trace))
    return math.acos(m_trace)
    
    
def get_rotation(matrix: np.ndarray) -> np.ndarray:
    ''' Get rotation matrix from the general transformation matrix.
        Args:
        matrix: The general transformation matrix 4x4.
    Returns:
        Rotation matrix 3x3. 
        '''
    return matrix[:3, :3]
    
def get_translation(matrix : np.ndarray) -> np.ndarray:
    ''' Get translation vector from the general transformation matrix.
        Args:
        matrix: The general transformation matrix 4x4.
    Returns:
        Translation vector. 
        '''
    return matrix[:3, 3]

def calculate_error(gt_transformation : np.ndarray, estimated_transformation : np.ndarray, degrees : bool = True):
    ''' Calculates the rotation and translation error.
        Args: 
        gt_transformation: Ground-truth transformation matrix 4x4.
        estimated_transformation: Estimated transformation matrix 4x4.
        degrees: flag whether return rotation error in radians or in degrees.
    Returns:
        Rotation and translation error. 
        '''
    gt_rotation, gt_translation = get_rotation(gt_transformation), get_translation(gt_transformation)
    est_rotation, est_translation = get_rotation(estimated_transformation), get_translation(estimated_transformation)
    translation_error = get_translation_error(gt_translation, est_translation)
    rotation_error = get_rotation_error(gt_rotation, est_rotation, degrees)
    return round(rotation_error, 2), round(translation_error, 2)


