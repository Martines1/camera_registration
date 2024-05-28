import argparse
import open3d as o3d
import torch
import numpy as np
import time
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error
import open3d as o3d
from config import make_cfg
from model import create_model


# def make_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--src_file", required=True, help="src point cloud numpy file")
#     parser.add_argument("--ref_file", required=True, help="src point cloud numpy file")
#     parser.add_argument("--gt_file", help="ground-truth transformation file")
#     parser.add_argument("--weights", required=True, help="model weights file")
#     return parser


def load_data(src_file, ref_file):
    
    src_points = np.asarray(src_file.points)
    ref_points = np.asarray(ref_file.points)
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }
    data_dict["transform"] = np.eye(4).astype(np.float32)
    return data_dict


def main(src_file, ref_file):    
    cfg = make_cfg()
    time_start = time.time()
    # prepare data
    data_dict = load_data(src_file, ref_file)
    #neighbor_limits = [380, 360, 360, 380]  # default setting in 3DMatch
    neighbor_limits = [1000, 100000, 1111, 1000]
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )
    weights= "models_dir/GeoTransformer/weights/geotransformer-3dmatch.pth.tar"
    # prepare model
    model = create_model(cfg).cpu()
    state_dict = torch.load(weights, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict["model"])
    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)
   
    estimated_transform = output_dict["estimated_transform"]
    
    
    return estimated_transform, time.time()-time_start
    

if __name__ == "__main__":
    main()
