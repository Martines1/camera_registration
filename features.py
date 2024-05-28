import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import cv2
import subprocess
import os

def extract_fpfh(src_pcd, tgt_pcd, VOXEL_SIZE):
    A_feats = extract(src_pcd, VOXEL_SIZE)
    B_feats = extract(tgt_pcd, VOXEL_SIZE)
    corrs_A, corrs_B = find_correspondences(
        A_feats, B_feats, mutual_filter=True)
    return corrs_A, corrs_B
    
    
def extract(pcd, voxel_size):
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return np.array(fpfh.data).T

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
  feat1tree = cKDTree(feat1)
  dists, nn_inds = feat1tree.query(feat0, k=knn)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True):
  nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
  corres01_idx0 = np.arange(len(nns01))
  corres01_idx1 = nns01

  if not mutual_filter:
    return corres01_idx0, corres01_idx1

  nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
  corres10_idx1 = np.arange(len(nns10))
  corres10_idx0 = nns10

  mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
  corres_idx0 = corres01_idx0[mutual_filter]
  corres_idx1 = corres01_idx1[mutual_filter]

  return corres_idx0, corres_idx1
    
    
    
def mutual_filter(first, second):
    existing_pairs = set()
    for i in range(len(second['matches'])):
        
        value = second['matches'][i]
        if value != -1:
            first_value = [ int(second['keypoints0'][i][0]), int(second['keypoints0'][i][1]) ]
            second_value = [ int(second['keypoints1'][value][0]), int(second['keypoints1'][value][1])]
            existing_pairs.add((str(first_value), str(second_value)))
            
    to_ignore = []
    for i in range(len(first['matches'])):
        value = first['matches'][i]
        if value != -1:
            first_value = [ int(first['keypoints0'][i][0]), int(first['keypoints0'][i][1]) ]
            second_value = [ int(first['keypoints1'][value][0]), int(first['keypoints1'][value][1])]
            to_find = (str(second_value), str(first_value))
            if to_find not in existing_pairs:
                to_ignore.append(i)
    return to_ignore
    
    
def get_zero(pcd):
    old_p = np.asarray(pcd.points)
    unique_points, counts = np.unique(old_p, axis=0, return_counts=True)
    most_occuring_index = np.argmax(counts)
    most_occuring_point = unique_points[most_occuring_index]
    return most_occuring_point
    
    
def superglue_pairs(src_pcd, tgt_pcd, src_pcd_raw, tgt_pcd_raw, limit, _mutual_filter = True):
    
    img_source = cv2.imread('images/source.png')
    img_target = cv2.imread('images/target.png')
    height_src, width_src, _ = img_source.shape
    height_tgt, width_tgt, _ = img_target.shape
    src_points = np.asarray(src_pcd.points)
    src_raw_points = np.asarray(src_pcd_raw.points).reshape((height_src, width_src, 3))
    tgt_points = np.asarray(tgt_pcd.points)
    tgt_raw_points = np.asarray(tgt_pcd_raw.points).reshape((height_tgt, width_tgt, 3))
    path = 'output_pairs/source_target_matches.npz'
    npz = np.load(path)
    corrs_A = []
    corrs_B = []
    to_ignore = []
    first_zero = get_zero(src_pcd_raw)
    second_zero = get_zero(tgt_pcd_raw)
    if _mutual_filter:
      to_ignore = mutual_filter(npz, np.load('output_pairs/target_source_matches.npz'))
    for i in range(len(npz['matches'])):
        value = npz['matches'][i]
        if(len(corrs_A) == limit):
            break
        if value != -1 and i not in to_ignore:
            first_value = [ int(npz['keypoints0'][i][0]), int(npz['keypoints0'][i][1]) ]
            first_point = src_raw_points[first_value[1]][first_value[0]]
            second_value = [ int(npz['keypoints1'][value][0]), int(npz['keypoints1'][value][1])]
            second_point = tgt_raw_points[second_value[1]][second_value[0]]
            if np.any(first_point != first_zero) and np.any(second_point != second_zero):
                index_A = np.where((src_points[:, 0] == first_point[0]) &
                                    (src_points[:, 1] == first_point[1]) &
                                    (src_points[:, 2] == first_point[2]))[0]             
                index_B = np.where((tgt_points[:, 0] == second_point[0]) &
                            (tgt_points[:, 1] == second_point[1]) &
                            (tgt_points[:, 2] == second_point[2]))[0]
                
                if len(index_B) != 0 and len(index_A) != 0:
                    corrs_A.append(index_A[0])
                    corrs_B.append(index_B[0])  
    return np.array(corrs_A), np.array(corrs_B)


def pdc_pairs(src_pcd, tgt_pcd, src_pcd_raw, tgt_pcd_raw, limit):
    img_source = cv2.imread('images/source.png')
    img_target = cv2.imread('images/target.png')
    height_src, width_src, _ = img_source.shape
    height_tgt, width_tgt, _ = img_target.shape
    src_points = np.asarray(src_pcd.points)
    src_raw_points = np.asarray(src_pcd_raw.points).reshape((height_src, width_src, 3))
    tgt_points = np.asarray(tgt_pcd.points)
    tgt_raw_points = np.asarray(tgt_pcd_raw.points).reshape((height_tgt, width_tgt, 3))
    curr_path = os.getcwd()
    command = f"conda run -n dense_matching_env  python {curr_path}/pdc_net/pdc_net_pair.py"     
    _ = subprocess.run(command, shell=True, capture_output=True)
    corrs_A = []
    corrs_B = []
    first_zero = get_zero(src_pcd_raw)
    second_zero = get_zero(tgt_pcd_raw)
    path = 'output_pairs/'
    first_cor = np.load(path+'source_target_0.npy')
    second_cor = np.load(path+'source_target_1.npy')
    for i in range(len(first_cor)):
        if(len(corrs_A) == limit):
            break
        first_value = [first_cor[i][0], first_cor[i][1]]
        first_point = src_raw_points[first_value[1]][first_value[0]]
        second_value = [second_cor[i][0], second_cor[i][1]]
        second_point = tgt_raw_points[second_value[1]][second_value[0]]
        if np.any(first_point != first_zero) and np.any(second_point != second_zero):
            index_A = np.where((src_points[:, 0] == first_point[0]) &
                                    (src_points[:, 1] == first_point[1]) &
                                    (src_points[:, 2] == first_point[2]))[0]
            index_B = np.where((tgt_points[:, 0] == second_point[0]) &
                        (tgt_points[:, 1] == second_point[1]) &
                        (tgt_points[:, 2] == second_point[2]))[0]
            if len(index_B) != 0 and len(index_A) != 0:
                corrs_A.append(index_A[0])
                corrs_B.append(index_B[0]) 
    return np.array(corrs_A), np.array(corrs_B)