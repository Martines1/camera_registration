import os, sys

import cv2

sys.path.append(os.getcwd()+'/models_dir/GeoTransformer')
import open3d as o3d
import numpy as np
import preprocessing
from error_calculation import calculate_error
import copy
from models_registration import Models
import features
import json
import argparse
from camera.camera_trigger import Device
from camera.gigev_camera_trigger import GIGEV_Device
from camera.checkPhoxi import checkPhoxi

class DeviceImage():
    def __init__(self, image, depth_map):
        self.image = image
        self.depth_map = depth_map


class PointCloud():
    def __init__(self, pcd):
        self.pcd = pcd
        self.pcd_raw = pcd
        self.number_of_points = np.asarray(pcd.points).shape[0]


class Register():
    """Class for registering source and target point clouds.

    Args:
        source_pcd (str): Path to the source point cloud file.
        target_pcd (str): Path to the target point cloud file.
        gt (str or None): Path to the ground truth transformation matrix file. Default is None.
        image_source (str or None): Path to the source image file. Default is None.
        image_target (str or None): Path to the target image file. Default is None.

    Attributes:
        source_pcd_raw (open3d.geometry.PointCloud): Raw source point cloud.
        target_pcd_raw (open3d.geometry.PointCloud): Raw target point cloud.
        source_pcd (open3d.geometry.PointCloud): Deep copy of the source point cloud.
        target_pcd (open3d.geometry.PointCloud): Deep copy of the target point cloud.
        gt (numpy.ndarray or None): Ground truth transformation matrix if provided, otherwise None.
        est (None): Estimated transformation matrix. Initialized as None.
        copy_of_source (open3d.geometry.PointCloud): Deep copy of the source point cloud for comparison.
        voxel_size (float): Voxel size for downsampling the point clouds. Default is 0.03.
        image_source (str or None): Path to the source image file. Default is None.
        image_target (str or None): Path to the target image file. Default is None.
        limit_of_texture_pairs (int): Limit of texture pairs for feature matching. Default is 1000.
        corrs_A (None): Placeholder for correspondence set A. Initialized as None.
        corrs_B (None): Placeholder for correspondence set B. Initialized as None.
    """

    def __init__(self, source_pcd: PointCloud, target_pcd: PointCloud, gt=None, image_source=None, image_target=None,
                 pairs_limit=1000, features="FPFH"):
        self.source = source_pcd
        self.target = target_pcd
        if gt is not None:
            self.gt = np.load(gt)
        else:
            self.gt = None
        self.est = None
        self.copy_of_source = copy.deepcopy(source_pcd.pcd)
        self.voxel_size = 0.03
        self.image_source = image_source
        self.image_target = image_target
        self.limit_of_texture_pairs = pairs_limit  # the default value for the limit of the texture pairs
        self.corrs_A = None
        self.corrs_B = None
        self.features = features

    def cpd_register(self):
        '''Performs registration using Coherent Point Drift.

        Args:
            cuda (bool): If True, uses CUDA acceleration. Default is False.

        Returns:
            None
        '''
        self.est, total_time = Models.cpd(copy.deepcopy(self.source.pcd), copy.deepcopy(self.target.pcd))
        print(f"CPD registration done! Total time: {round(total_time, 2)} sec.")
        self.source.pcd_raw.transform(self.est)
        self._calculate_error()

    def geotransformer_register(self):
        '''Performs registration using Geometric Transformer.

        Returns:
            None
        '''
        self.est, total_time = Models.geotransformer(copy.deepcopy(self.source.pcd), copy.deepcopy(self.target.pcd))
        print(f"GeoTransformer registration done! Total time: {round(total_time, 2)} sec.")
        self.source.pcd_raw.transform(self.est)
        self._calculate_error()

    def teaser_register(self):
        '''Performs Teaser++ registration.

        Args:
            feat (str): Type of features used for registration. 
                        Supported values are "FPFH" for geometric or 
                        "Superglue" or "PDC-Net+" for texture.

        Returns:
            None
        '''
        if self.features == "FPFH":
            self._get_features(self.features)
        else:
            self.voxel_size = 1
            self._get_features(self.features)
        if self.corrs_A is None:  # if corrs_A is None, then also corrs_B must be None.
            return
        self.est, total_time = Models.teaser(copy.deepcopy(self.source.pcd), copy.deepcopy(self.target.pcd),
                                             self.voxel_size, self.corrs_A, self.corrs_B)
        print(f"Teaser++ registration done! Total time: {round(total_time+self.features_time, 2)} sec.")
        self.source.pcd_raw.transform(self.est)
        self._calculate_error()
        self.corrs_A, self.corrs_B = None, None

    def gcnet_register(self):
        '''Performs registration using Geometry-guided Consistent.

        Returns:
            None
        '''
        self.est, total_time = Models.gcnet(copy.deepcopy(self.source.pcd), copy.deepcopy(self.target.pcd))
        print(f"GCNet registration done! Total time: {round(total_time, 2)} sec.")
        self.source.pcd_raw.transform(self.est)
        self._calculate_error()

    def mac_register(self):
        '''Performs registration using Maximal Cliques.

        Args:
            feat (str): Type of features used for registration. 
                        Supported values are "FPFH" for geometric or 
                        "Superglue" or "PDC-Net+" for texture.

        Returns:
            None
        '''
        if self.features == "FPFH":
            self.corrs_A, self.corrs_B = None, None
        else:
            self.voxel_size = 1
            self._get_features(self.features)
            if self.corrs_A is None:  # if corrs_A is None, then also corrs_B must be None.
                return
        self.est, total_time = Models.mac(copy.deepcopy(self.source.pcd), copy.deepcopy(self.target.pcd), self.corrs_A,
                                          self.corrs_B)
        print(f"MAC registration done! Total time: {round(total_time + self.features_time, 2)} sec.")
        self.source.pcd_raw.transform(self.est)
        self._calculate_error()

    def pointdsc_register(self):
        '''Performs registration using Deep Spatial Consistency.

        Args:
            feat (str): Type of features used for registration. 
                        Supported values are "FPFH" for geometric or 
                        "Superglue" or "PDC-Net+" for texture.

        Returns:
            None
        '''
        if self.features == "FPFH":
            self.corrs_A, self.corrs_B = None, None
        else:
            self.voxel_size = 1
            self._get_features(self.features)
            if self.corrs_A is None:  # if corrs_A is None, then also corrs_B must be None.
                return
        self.est, total_time = Models.pointdsc(copy.deepcopy(self.source.pcd), copy.deepcopy(self.target.pcd),
                                               self.corrs_A, self.corrs_B)
        print(f"PointDSC registration done! Total time: {round(total_time + self.features_time, 2)} sec.")
        self.source.pcd_raw.transform(self.est)
        self._calculate_error()

    def _get_features(self, feat: str):
        '''
        Save geometric or texture features into self.corrs_A, self.corrs_B.
        
        Returns:
            None
        '''
        if self.corrs_A is None:
            if feat == "FPFH":
                self.corrs_A, self.corrs_B = features.extract_fpfh(self.source.pcd, self.target.pcd, self.voxel_size)
                self.voxel_size = 0.02  # especially for teaser++
            elif feat == "PDC-Net+":
                if self.image_source is not None and self.image_target is not None:
                    self.corrs_A, self.corrs_B, self.features_time = features.pdc_pairs(self.source, self.target,
                                                                    self.limit_of_texture_pairs, self.image_source,
                                                                    self.image_target)
                else:
                    print("Could not find images!")
            else:
                print("Wrong features! Available features: FPFH and PDC-Net+")

    def _calculate_error(self):
        ''' Calculates rotation and translation error and print them.
        
        Returns:
            None
        '''
        if self.est is not None and self.gt is not None:
            self.deg_e, self.tran_e = calculate_error(self.gt, self.est)
            print("Rotation error:\033[96m {}\033[00m, translation error:\033[96m {}\033[00m".format(self.deg_e,
                                                                                                     self.tran_e))
        else:
            print("No ground-truth value provided!")

    def icp_refinement(self, threshold: float):
        '''Refines the alignment using Iterative Closest Point (ICP) algorithm after initial registration.

        Args:
            threshold (float): Threshold for convergence in the ICP algorithm.

        Returns:
            open3d.geometry.PointCloud: The refined point cloud after ICP refinement.
        '''
        if self.est is not None:
            icp_reg = o3d.pipelines.registration.registration_icp(
                self.source.pcd, self.target.pcd, threshold, self.est,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
            if self.gt is not None:
                temp_deg_e, temp_tran_e = calculate_error(self.gt, icp_reg.transformation)
                print(
                    f"Without ICP errors: rot {self.deg_e}, tran {self.tran_e} \nWith ICP errors: rot {temp_deg_e}, tran {temp_tran_e}")
            self.source.pcd_raw.transform(icp_reg.transformation)


    def visualize_result(self):
        ''' Visualize the source and target point clouds.
        '''
        o3d.visualization.draw_geometries([self.source.pcd_raw, self.target.pcd_raw])

    def compare(self, colors = False):
        ''' Compare before and after registration.\\
        Red is the source point cloud before registration.\\
        Green is the target point cloud.\\
        Blue is the source point cloud after registration.
        
        Returns:
            None
        '''
        if colors:
            self.copy_of_source.paint_uniform_color([1, 0, 0])
            self.target.pcd_raw.paint_uniform_color([0, 1, 0])
            self.source.pcd_raw.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([self.copy_of_source, self.target.pcd_raw])
        o3d.visualization.draw_geometries([self.source.pcd_raw, self.target.pcd_raw])

    def get_pcd(self):
        '''Returns the source and target point clouds.

        Returns:
            tuple: A tuple containing the source and target point clouds.
        '''
        return self.source.pcd_raw, self.target.pcd_raw

    def preprocess(self, undefined_data=False, scale_factor=None, voxel_size=None, preserve_edges=True):
        '''Preprocesses the source and target point clouds.

        Args:
            undefined_data (bool): If True, removes undefined data points from the point clouds.
            scale_factor (float): Scale factor to apply to the point clouds.
            voxel_size (float): Voxel size for downsampling the point clouds.
            preserve_edges (bool): If True, preserves edges in the point clouds after downsampling.

        Returns:
            None
        '''
        if undefined_data:
            self.source.pcd = preprocessing.remove_undefined(self.source.pcd)
            self.target.pcd = preprocessing.remove_undefined(self.target.pcd)

        if scale_factor is not None:
            preprocessing.scale_pcd(self.source.pcd, scale_factor)
            preprocessing.scale_pcd(self.target.pcd, scale_factor)
        if voxel_size is not None:
            self.voxel_size = voxel_size
            self.source.pcd = preprocessing.downsample_pcd(self.source.pcd, voxel_size)
            self.target.pcd = preprocessing.downsample_pcd(self.target.pcd, voxel_size)
            if preserve_edges and self.image_source is not None and self.image_target is not None:
                self.source.pcd = preprocessing.canny_edge(self.source.pcd, self.source.pcd_raw, scale_factor,
                                                           self.image_source)
                self.target.pcd = preprocessing.canny_edge(self.target.pcd, self.target.pcd_raw, scale_factor,
                                                           self.image_target)
        o3d.visualization.draw_geometries([self.source.pcd, self.target.pcd])

    def get_est(self):
        return self.est


def run(src_image, src_pcd, tgt_image, tgt_pcd):
    with open('config.json', 'r') as file:
        data = json.load(file)
    data = data.get("registration")
    chosen_features = ["FPFH", "PDC-Net+"][int(data["features"])]
    t = Register(src_pcd, tgt_pcd, gt=None, image_source=src_image, image_target=tgt_image, features=chosen_features,
                 pairs_limit=int(data["pairs_limit"]))
    if eval(data["preprocessing"]):
        t.preprocess(eval(data["undefined_data"]), float(data["scale_factor"]), float(data["voxel_size"]),
                     eval(data["preserve_edges"]))
    chosen_method = [t.cpd_register, t.geotransformer_register, t.teaser_register, t.gcnet_register, t.mac_register,
                     t.pointdsc_register][int(data["model"])]
    chosen_method()
    t.compare()
    return t.est
    #t.icp_refinement(0.1)
    #t.compare()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some device IDs.")
    parser.add_argument('--device_id', nargs='+', help='List of device IDs', required=True)
    args = parser.parse_args()
    device_ids = args.device_id
    use_phoxi = checkPhoxi()
    print("Loaded devices: {}".format(device_ids))
    if use_phoxi:
        D = Device(device_ids)
    else:
        D = GIGEV_Device(device_ids)
    D.connect()
    D.trigger()
    pcd, img, depth = D.results(0)
    source_pcd, source_image = PointCloud(pcd), DeviceImage(img, depth)
    for i in range(1, len(device_ids)):
        pcd, image, depth = D.results(i)
        other_pcd, other_image = PointCloud(pcd), DeviceImage(image, depth)
        print("Registration in the process..")
        est = run(other_image, other_pcd, source_image, source_pcd)
        if use_phoxi:
            D.set_transformation_matrix(est, i)
    print("Registration process done!")
    while cv2.waitKey(0) != 42:
        continue


