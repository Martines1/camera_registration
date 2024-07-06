import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import cv2
import subprocess
import os
from models_dir.PDC_NET.pdc_net_pair import get_pair


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
    corres10_idx0 = nns10

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1


def pdc_pairs(src_pcd, tgt_pcd, limit, img_source, img_target):
    height_src, width_src, _ = img_source.image.shape
    height_tgt, width_tgt, _ = img_target.image.shape
    src_points = np.asarray(src_pcd.pcd.points)
    src_raw_points = np.asarray(src_pcd.pcd_raw.points).reshape((height_src, width_src, 3))
    tgt_points = np.asarray(tgt_pcd.pcd.points)
    tgt_raw_points = np.asarray(tgt_pcd.pcd_raw.points).reshape((height_tgt, width_tgt, 3))
    curr_time, first_cor, second_cor = get_pair(img_source.image, img_target.image, img_source.depth_map,
                                                img_target.depth_map, limit)
    corrs_A = []
    corrs_B = []
    for i in range(len(first_cor)):
        first_value = [first_cor[i][0], first_cor[i][1]]
        first_point = src_raw_points[first_value[1]][first_value[0]]
        second_value = [second_cor[i][0], second_cor[i][1]]
        second_point = tgt_raw_points[second_value[1]][second_value[0]]
        index_A = np.where((src_points[:, 0] == first_point[0]) &
                           (src_points[:, 1] == first_point[1]) &
                           (src_points[:, 2] == first_point[2]))[0]
        index_B = np.where((tgt_points[:, 0] == second_point[0]) &
                           (tgt_points[:, 1] == second_point[1]) &
                           (tgt_points[:, 2] == second_point[2]))[0]
        if len(index_B) != 0 and len(index_A) != 0:
            corrs_A.append(index_A[0])
            corrs_B.append(index_B[0])
    return np.array(corrs_A), np.array(corrs_B), curr_time
