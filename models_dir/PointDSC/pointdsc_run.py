import json
import copy
import argparse
import os
import sys
sys.path.append('models_dir/PointDSC/')
from easydict import EasyDict as edict
from models.PointDSC import PointDSC
from utils.pointcloud import estimate_normal
import torch
import numpy as np
import open3d as o3d 
import time



def extract_fpfh_features(pcd, downsample, device):
    raw_src_pcd = copy.deepcopy(pcd)
    estimate_normal(raw_src_pcd, radius=downsample*2)
    src_pcd = raw_src_pcd.voxel_down_sample(downsample)
    src_features = o3d.pipelines.registration.compute_fpfh_feature(src_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=downsample * 5, max_nn=100))
    src_features = np.array(src_features.data).T
    src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
    return raw_src_pcd, np.array(src_pcd.points), src_features

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not source_temp.has_normals():
        estimate_normal(source_temp)
        estimate_normal(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def register(src, tgt, feat_A = None, feat_B = None):
    start_time = time.time()
    config_path = f'models_dir/PointDSC/snapshot/PointDSC_3DMatch_release/config.json' 
    config = json.load(open(config_path, 'r'))
    config = edict(config)
    use_gpu = False
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = PointDSC(
        in_dim=config.in_dim,
        num_layers=config.num_layers,
        num_channels=config.num_channels,
        num_iterations=config.num_iterations,
        ratio=config.ratio,
        sigma_d=config.sigma_d,
        k=config.k,
        nms_radius=config.inlier_threshold,
    ).to(device)
    model.eval()
    # extract features
    if feat_A is None:
        raw_src_pcd, src_pts, src_features = extract_fpfh_features(src, config.downsample, device)
        raw_tgt_pcd, tgt_pts, tgt_features = extract_fpfh_features(tgt, config.downsample, device)
        distance = np.sqrt(2 - 2 * (src_features @ tgt_features.T) + 1e-6)
        source_idx = np.argmin(distance, axis=1)
        corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
        src_keypts = src_pts[corr[:,0]]
        tgt_keypts = tgt_pts[corr[:,1]]
    
    else:
        raw_src_pcd = copy.deepcopy(src)

        raw_tgt_pcd = copy.deepcopy(tgt)
        src_pts = np.asarray(raw_src_pcd.points)
        tgt_pts = np.asarray(raw_tgt_pcd.points)
        corr = []
        for i in range(len(feat_A)):
            corr.append([feat_A[i], feat_B[i]]) 
        corr = np.array(corr)
        src_keypts = src_pts[feat_A]
        tgt_keypts = tgt_pts[feat_B]
    # matching
    corr_pos = np.concatenate([src_keypts, tgt_keypts], axis=-1)
    corr_pos = corr_pos - corr_pos.mean(0)
    # outlier rejection
    data = {
            'corr_pos': torch.from_numpy(corr_pos)[None].to(device).float(),
            'src_keypts': torch.from_numpy(src_keypts)[None].to(device).float(),
            'tgt_keypts': torch.from_numpy(tgt_keypts)[None].to(device).float(),
            'testing': True,
            }
    res = model(data)
    end_time = time.time() - start_time
    est = res['final_trans'][0].detach().cpu().numpy()
    return est, end_time
    