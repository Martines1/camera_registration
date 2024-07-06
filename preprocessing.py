import numpy as np
import open3d as o3d
import cv2

def remove_undefined(pcd):
    ''' Remove undefined points in the point cloud.
        Args:
        pcd: The point cloud.
    Returns:
        Point cloud without undefined points. 
        '''
    old_p = np.asarray(pcd.points)
    old_c = np.asarray(pcd.colors)
    new_p = []
    new_c = []
    unique_points, counts = np.unique(old_p, axis=0, return_counts=True)
    most_occuring_index = np.argmax(counts)
    most_occuring_point = unique_points[most_occuring_index]
    for i,p in enumerate(old_p):
        if np.any(p != most_occuring_point):
            new_p.append(p)
            if old_c.shape[0] != 0:
                new_c.append(old_c[i])
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(np.asarray(new_p))
    if len(new_c) != 0:
        result.colors = o3d.utility.Vector3dVector(np.asarray(new_c))
    return result

def downsample_pcd(pcd, v_size):
    ''' Voxel downsampling.
        Args:
        pcd: The point cloud.
        v_size: Size of the voxel.
    Returns:
        Downsampled point cloud. 
        '''
    return pcd.voxel_down_sample(voxel_size=v_size)

def scale_pcd(pcd, scale_factor):
    ''' Isotropic scaling of the point cloud.
        Args:
        pcd: The point cloud.
        scale_factor: The scale factor.
    Returns:
        None. But the point cloud will remain scaled. 
        '''
    pcd.scale(scale_factor, center=(0, 0, 0))

def get_zero(pcd):
    old_p = np.asarray(pcd.points)
    unique_points, counts = np.unique(old_p, axis=0, return_counts=True)
    most_occuring_index = np.argmax(counts)
    most_occuring_point = unique_points[most_occuring_index]
    return most_occuring_point

def canny_edge(pcd, pcd_raw, _scale_factor, input_img):
    ''' Preserve edges after downsampling.
    Args:
    pcd: The downsampled point cloud.
    Returns:
        Point cloud with preserved edges.
    '''
    scale_factor = 1 if _scale_factor is None else _scale_factor
    img = cv2.imread(input_img)
    height, width, _  = img.shape
    points_raw = np.asarray(pcd_raw.points).reshape((height, width, 3))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    indices = np.where(edges != 0)
    points = np.asarray(pcd.points)
    zero = get_zero(pcd_raw)
    new_points = np.asarray(pcd.points)
    for y, x in zip(indices[0], indices[1]):
        new_p = points_raw[y][x]
        if np.all(new_p == zero):
            continue
        new_p *= scale_factor
        index = np.where((new_p[0] == points[:, 0]) & 
                         (new_p[1] == points[:, 1]) &
                        (new_p[2] == points[:, 2]))[0]
        if len(index) == 0:
            new_points = np.append(new_points, new_p.reshape(1, -1), axis=0)
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(new_points)
    return result
    