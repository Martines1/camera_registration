import numpy as np
import open3d as o3d


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
            new_c.append(old_c[i])
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(np.asarray(new_p))
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

