import os
import sys
import sys
sys.stderr = open(os.devnull, "w")
import time
import torch
import numpy as np
import torch
import imageio.v2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
sys.path.append(os.getcwd()+'/models_dir/PDC_NET')
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from model_selection import select_model
torch.set_grad_enabled(False)
from utils_flow.visualization_utils import make_sparse_matching_plot
from models.inference_utils import estimate_mask
from utils_flow.flow_and_mapping_operations import convert_flow_to_mapping
from validation.utils import matches_from_flow
from admin.stats import DotDict

def get_pair(src_img, tgt_img, depth_map_src, depth_map_tgt, limit):
    now = time.time()
    current_dir = os.getcwd() + '/models_dir/PDC_NET'
    cv2.imwrite(current_dir + '/source.png', src_img)
    cv2.imwrite(current_dir + '/target.png', tgt_img)
    query_image = imageio.v2.imread(current_dir + '/source.png', mode='RGB')
    reference_image = imageio.v2.imread(current_dir + '/target.png', mode='RGB')
    query_image_shape = query_image.shape
    ref_image_shape = reference_image.shape


    model = 'PDCNet_plus'
    pre_trained_model = 'megadepth'
    global_optim_iter = 3
    local_optim_iter = 7
    path_to_pre_trained_models = current_dir + '/pre_trained_models/'

    # inference parameters for PDC-Net
    network_type = model  # will only use these arguments if the network_type is 'PDCNet' or 'PDCNet_plus'
    multi_stage_type = 'h'
    confidence_map_R =1
    ransac_thresh = 0.1
    mask_type = 'proba_interval_1_above_10'  # for internal homo estimation
    homography_visibility_mask = True
    scaling_factors = [0.5, 0.6, 0.88, 1, 1.33, 1.66, 2]
    compute_cyclic_consistency_error = True  # here to compare multiple uncertainty
    save_ = 'output'
    # usually from argparse
    args = DotDict({'network_type': network_type, 'multi_stage_type': multi_stage_type, 'confidence_map_R': confidence_map_R,
                    'ransac_thresh': ransac_thresh, 'mask_type': mask_type,
                    'homography_visibility_mask': homography_visibility_mask, 'scaling_factors': scaling_factors,
                    'compute_cyclic_consistency_error': compute_cyclic_consistency_error, 'save_dir': save_, 'resize': -1})

    network, estimate_uncertainty = select_model(
        model, pre_trained_model, args, global_optim_iter, local_optim_iter,
        path_to_pre_trained_models=path_to_pre_trained_models)
    estimate_uncertainty = True

    # convert the images to correct format to be processed by the network: torch Tensors, format B, C, H, W.
    # pad both images to the same size, to be processed by network
    #query_image_, reference_image_ = pad_to_same_shape(query_image, reference_image)
    # convert numpy to torch tensor and put it in right format
    query_image_ = torch.from_numpy(query_image).permute(2, 0, 1).unsqueeze(0)
    reference_image_ = torch.from_numpy(reference_image).permute(2, 0, 1).unsqueeze(0)

    if estimate_uncertainty:
        estimated_flow, uncertainty_components = network.estimate_flow_and_confidence_map(query_image_, reference_image_)
    else:
        if args.flipping_condition and 'GLUNet' in args.model:
            estimated_flow = network.estimate_flow_with_flipping_condition(query_image_, reference_image_,
                                                                        mode='channel_first')
        else:
            estimated_flow = network.estimate_flow(query_image_, reference_image_, mode='channel_first')
    # removes the padding
    estimated_flow = estimated_flow[:, :, :ref_image_shape[0], :ref_image_shape[1]]

    # confidence estimation + visualization
    if not estimate_uncertainty:
        raise ValueError

    uncertainty_key = 'p_r'  # 'inv_cyclic_consistency_error'
    #'p_r', 'inv_cyclic_consistency_error' can also be used as a confidence measure
    # 'cyclic_consistency_error' can also be used, but that's an uncertainty measure
    confidence_map = uncertainty_components[uncertainty_key]
    confidence_map = confidence_map[:, :, :ref_image_shape[0], :ref_image_shape[1]]

    color = [255, 102, 51]

    # get the mask according to uncertainty estimation
    mask_type = 'proba_interval_1_above_10' # 'cyclic_consistency_error_below_2'
    #'proba_interval_1_above_10' can be used in association with all networks, relying on p_r for PDC-Net
    # and inv_cyclic_consistency_error for the other networks
    # Alternatively, can use instead 'cyclic_consistency_error_below_x'
    # choices_for_mask_type = ['cyclic_consistency_error_below_x', 'x_percent_most_certain', 'variance_below_x',
    #                          'proba_interval_z_above_x_NMS_y',  'proba_interval_z_above_x_grid_y',
    #                          'proba_interval_z_above_x']  x, y and z are numbers to choose


    mask_padded = estimate_mask(mask_type, uncertainty_components)
    if 'warping_mask' in list(uncertainty_components.keys()):
        # get mask from internal multi stage alignment, if it took place
        mask_padded = uncertainty_components['warping_mask'] * mask_padded

    # remove the padding
    mask = mask_padded[:, :ref_image_shape[0], :ref_image_shape[1]]

    # remove point that lead to outside the query image
    mapping_estimated = convert_flow_to_mapping(estimated_flow)
    mask = mask & mapping_estimated[:, 0].ge(0) & mapping_estimated[:, 1].ge(0) & \
    mapping_estimated[:, 0].le(query_image_shape[1] - 1) & mapping_estimated[:, 1].le(query_image_shape[0] - 1)

    mkpts_query, mkpts_ref = matches_from_flow(estimated_flow, mask)

    confidence_values = confidence_map.squeeze()[mask.squeeze()].cpu().numpy()
    sort_index = np.argsort(np.array(confidence_values)).tolist()[::-1]  # from highest to smallest
    confidence_values = np.array(confidence_values)[sort_index]
    mkpts_query = np.array(mkpts_query)[sort_index]
    mkpts_ref = np.array(mkpts_ref)[sort_index]

    if len(mkpts_query) < 5:
        mkpts_query = np.empty([0, 2], dtype=np.float32)
        mkpts_ref = np.empty([0, 2], dtype=np.float32)
        confidence_values = np.empty([0], dtype=np.float32)
    total_number_of_corr = len(confidence_values)


    top_20 = int(total_number_of_corr * 0.3)
    mkpts_q = np.round(mkpts_query[:top_20]).astype(int)
    mkpts_r = np.round(mkpts_ref[:top_20]).astype(int)
    confidence_values = confidence_values[:top_20]
    # undefined_src = get_undefined(depth_map_src)
    # undefined_tgt = get_undefined(depth_map_tgt)
    filtered_mkpts_q = []
    filtered_mkpts_r = []
    filtered_confidence_values = []
    for i, (q, r) in enumerate(zip(mkpts_q, mkpts_r)):
        x_q, y_q = q
        x_r, y_r = r
        if depth_map_src[y_q, x_q] != 0 and depth_map_tgt[y_r, x_r] != 0:
            filtered_mkpts_q.append([x_q, y_q])
            filtered_mkpts_r.append([x_r, y_r])
            filtered_confidence_values.append(confidence_values[i])

    every_nth = 1 if len(filtered_mkpts_q) <= limit else round(len(filtered_mkpts_q) / limit)

    # Convert lists back to numpy arrays
    mkpts0 = np.array(filtered_mkpts_q)[::every_nth]
    mkpts1 = np.array(filtered_mkpts_r)[::every_nth]
    confidence_values = np.array(filtered_confidence_values)[::every_nth]

    #mkpts0, mkpts1 = np.round(mkpts_q[:top_20:100]).astype(int), np.round(mkpts_r[:top_20:100]).astype(int)
    color = cm.jet(confidence_values)
    out = make_sparse_matching_plot(query_image, reference_image, mkpts0, mkpts1, color, margin=10)


    plt.imsave(f"output/result.png", out)
    return time.time() - now, mkpts0, mkpts1
