import open3d as o3d
import numpy as np
import preprocessing
from error_calculation import calculate_error
import copy
from models_registration import Models
import features
import cv2

class Register():
    '''Class for registering source and target point clouds.

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
    '''
    def __init__(self, source_pcd, target_pcd, gt = None, image_source = None, image_target = None):
        self.source_pcd_raw = o3d.io.read_point_cloud(source_pcd)
        self.target_pcd_raw = o3d.io.read_point_cloud(target_pcd)
        self.source_pcd = copy.deepcopy(self.source_pcd_raw)
        self.target_pcd = copy.deepcopy(self.target_pcd_raw)
        if gt is not None:
            self.gt = np.load(gt)
        else:
            self.gt = None
        self.est = None
        self.copy_of_source = copy.deepcopy(self.source_pcd_raw)
        self.voxel_size = 0.03
        self.image_source = image_source
        self.image_target = image_target
        self.limit_of_texture_pairs = 1000 # the default value for the limit of the texture pairs
        self.corrs_A = None
        self.corrs_B = None

    def cpd_register(self, cuda : bool = False):
        '''Performs registration using Coherent Point Drift.

        Args:
            cuda (bool): If True, uses CUDA acceleration. Default is False.

        Returns:
            None
        '''
        self.est, total_time = Models.cpd(copy.deepcopy(self.source_pcd), copy.deepcopy(self.target_pcd), cuda)
        print(f"CPD registration done! Total time: {round(total_time,2)} sec.")
        self.source_pcd_raw.transform(self.est)
        self._calculate_error()
            
        
    def geotransformer_register(self):
        '''Performs registration using Geometric Transformer.

        Returns:
            None
        '''
        self.est, total_time = Models.geotransformer(copy.deepcopy(self.source_pcd), copy.deepcopy(self.target_pcd))
        print(f"GeoTransformer registration done! Total time: {round(total_time,2)} sec.")
        self.source_pcd_raw.transform(self.est)
        self._calculate_error()

    def teaser_register(self, feat : str):
        '''Performs Teaser++ registration.

        Args:
            feat (str): Type of features used for registration. 
                        Supported values are "FPFH" for geometric or 
                        "Superglue" or "PDC-Net+" for texture.

        Returns:
            None
        '''
        if feat == "FPFH":
            self._get_features(feat)
        else:
            self.voxel_size = 1
            self._get_features(feat)
        if self.corrs_A is None: # if corrs_A is None, then also corrs_B must be None.
            return
        self.est, total_time = Models.teaser(copy.deepcopy(self.source_pcd), copy.deepcopy(self.target_pcd), self.voxel_size, self.corrs_A, self.corrs_B)
        print(f"Teaser++ registration done! Total time: {round(total_time,2)} sec.")
        self.source_pcd_raw.transform(self.est)
        self._calculate_error()
        self.corrs_A, self.corrs_B = None, None

    def gcnet_register(self):
        '''Performs registration using Geometry-guided Consistent.

        Returns:
            None
        '''
        self.est, total_time = Models.gcnet(copy.deepcopy(self.source_pcd), copy.deepcopy(self.target_pcd))
        print(f"GCNet registration done! Total time: {round(total_time,2)} sec.")
        self.source_pcd_raw.transform(self.est)
        self._calculate_error()

    def mac_register(self, feat : str):
        '''Performs registration using Maximal Cliques.

        Args:
            feat (str): Type of features used for registration. 
                        Supported values are "FPFH" for geometric or 
                        "Superglue" or "PDC-Net+" for texture.

        Returns:
            None
        '''
        if feat == "FPFH":
           self.corrs_A, self.corrs_B = None, None
        else:
            self.voxel_size = 1
            self._get_features(feat)
            if self.corrs_A is None: # if corrs_A is None, then also corrs_B must be None.
                return
        self.est, total_time = Models.mac(copy.deepcopy(self.source_pcd), copy.deepcopy(self.target_pcd), self.corrs_A, self.corrs_B)
        print(f"MAC registration done! Total time: {round(total_time,2)} sec.")
        self.source_pcd_raw.transform(self.est)
        self._calculate_error()

    def pointdsc_register(self, feat : str):
        '''Performs registration using Deep Spatial Consistency.

        Args:
            feat (str): Type of features used for registration. 
                        Supported values are "FPFH" for geometric or 
                        "Superglue" or "PDC-Net+" for texture.

        Returns:
            None
        '''
        if feat == "FPFH":
           self.corrs_A, self.corrs_B = None, None
        else:
            self.voxel_size = 1
            self._get_features(feat)
            if self.corrs_A is None: # if corrs_A is None, then also corrs_B must be None.
                return
        self.est, total_time = Models.pointdsc(copy.deepcopy(self.source_pcd), copy.deepcopy(self.target_pcd), self.corrs_A, self.corrs_B)
        print(f"PointDSC registration done! Total time: {round(total_time,2)} sec.")
        self.source_pcd_raw.transform(self.est)
        self._calculate_error()


    def _get_features(self, feat : str):
        '''
        Save geometric or texture features into self.corrs_A, self.corrs_B.
        
        Returns:
            None
        '''
        if self.corrs_A is None:
            if feat == "FPFH":
                self.corrs_A, self.corrs_B =  features.extract_fpfh(self.source_pcd, self.target_pcd, self.voxel_size)
                self.voxel_size = 0.02 # especially for teaser++
            elif feat== "Superglue":
                if self.image_source is not None and self.image_target is not None:
                    cv2.imwrite("images/source.png", cv2.imread(self.image_source))
                    cv2.imwrite("images/target.png", cv2.imread(self.image_target))
                    self.corrs_A, self.corrs_B =  features.superglue_pairs(self.source_pcd, self.target_pcd, self.source_pcd_raw, self.target_pcd_raw, self.limit_of_texture_pairs)
                
                else:
                    print("Could not find images!")
            elif feat== "PDC-Net+":
                if self.image_source is not None and self.image_target is not None:
                    cv2.imwrite("images/source.png", cv2.imread(self.image_source))
                    cv2.imwrite("images/target.png", cv2.imread(self.image_target))
                    self.corrs_A, self.corrs_B =  features.pdc_pairs(self.source_pcd, self.target_pcd, self.source_pcd_raw, self.target_pcd_raw, self.limit_of_texture_pairs)
                else:
                    print("Could not find images!")
            else:
                print("Wrong features! Available features: FPFH, Superglue, PDC-Net+")

    

    def _calculate_error(self):
        ''' Calculates rotation and translation error and print them.
        
        Returns:
            None
        '''
        if self.est is not None and self.gt is not None:
            self.deg_e, self.tran_e = calculate_error(self.gt, self.est)
            print("Rotation error:\033[96m {}\033[00m, translation error:\033[96m {}\033[00m" .format(self.deg_e, self.tran_e))
        else:
            print("No ground-truth value provided!")
    
    
    def icp_refinement(self, threshold : float):
        '''Refines the alignment using Iterative Closest Point (ICP) algorithm after initial registration.

        Args:
            threshold (float): Threshold for convergence in the ICP algorithm.

        Returns:
            open3d.geometry.PointCloud: The refined point cloud after ICP refinement.
        '''
        if self.est is not None:
            icp_reg = o3d.pipelines.registration.registration_icp(
            self.source_pcd, self.target_pcd, threshold, self.est,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)) 
            if self.gt is not None:
                temp_deg_e, temp_tran_e = calculate_error(self.gt, icp_reg.transformation)
                print(f"Without ICP errors: rot {self.deg_e}, tran {self.tran_e} \nWith ICP errors: rot {temp_deg_e}, tran {temp_tran_e}")                 
            icp_pcd = copy.deepcopy(self.source_pcd_raw).transform(icp_reg.transformation)
            return icp_pcd
        
    def visualize_result(self):
        ''' Visualize the source and target point clouds.
        '''
        o3d.visualization.draw_geometries([self.source_pcd_raw, self.target_pcd_raw])

    def compare(self):
        ''' Compare before and after registration.\\
        Red is the source point cloud before registration.\\
        Green is the target point cloud.\\
        Blue is the source point cloud after registration.
        
        Returns:
            None
        '''
        self.copy_of_source.paint_uniform_color([1, 0, 0])
        self.target_pcd_raw.paint_uniform_color([0, 1, 0])
        self.source_pcd_raw.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([self.copy_of_source, self.target_pcd_raw])
        o3d.visualization.draw_geometries([self.source_pcd_raw, self.target_pcd_raw])

    
    def get_pcd(self):
        '''Returns the source and target point clouds.

        Returns:
            tuple: A tuple containing the source and target point clouds.
        '''
        return self.source_pcd_raw, self.target_pcd_raw

    def preprocess(self, undefined_data = False, scale_factor=None, voxel_size=None, preserve_edges = True):
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
            self.source_pcd = preprocessing.remove_undefined(self.source_pcd)
            self.target_pcd = preprocessing.remove_undefined(self.target_pcd)
            
        if scale_factor is not None:
            preprocessing.scale_pcd(self.source_pcd, scale_factor)
            preprocessing.scale_pcd(self.target_pcd, scale_factor)
        if voxel_size is not None:
            self.voxel_size = voxel_size
            self.source_pcd = preprocessing.downsample_pcd(self.source_pcd, voxel_size)
            self.target_pcd = preprocessing.downsample_pcd(self.target_pcd, voxel_size)
            if preserve_edges and self.image_source is not None and self.image_target is not None:
                self.source_pcd = preprocessing.canny_edge(self.source_pcd, self.source_pcd_raw, scale_factor, self.image_source)
                self.target_pcd = preprocessing.canny_edge(self.target_pcd, self.target_pcd_raw, scale_factor, self.image_target)

    def clear_correspondences(self):
        '''
        Set correspondences to None.
        
        Returns:
            None
        '''
        self.corrs_A, self.corrs_B = None, None

    def save_est(self):
        '''Saves the estimated transformation matrix to a file.

        If the estimated transformation matrix exists, it saves it to a file named "estimated_transformation.npy".

        Returns:
            None
        '''
        if self.est is not None:
            np.save("estimated_transformation.npy", self.est)
            print("Estimated transformation was successfully saved!")
        else:
            print("Could not find estimated transformation!")

    def demo_transformation_texture(self, feat : str = "Superglue"):
        '''Demonstrates texture-based registration with custom transformation.

        This method prepares the point clouds for texture-based registration using PDC-Net+ features.
        It scales both the source and target point clouds, applies a custom transformation matrix to the source,
        and updates the ground truth transformation matrix accordingly.
        Args:
            feat (str): Type of texture features used for registration. 
                        Supported values are 
                        "Superglue" or "PDC-Net+".
        Returns:
            None
        '''
        if feat != "Superglue" and feat != "PDC-Net+":
            print("Wrong features!")
            return
        self._get_features(feat)
        self.source_pcd.scale(0.0025, center= (0, 0, 0))
        self.target_pcd.scale(0.0025, center=(0, 0, 0))
        th = np.deg2rad(30.0)
        tr =np.array([[np.cos(th), -np.sin(th), 0.0, 250.0],
                            [np.sin(th), np.cos(th), 0.0, 65.0],
                            [0, 0, 1.0, 30.0],
                            [0.0, 0.0, 0.0, 1.0]])
        tr[:3, 3] *= 0.0025 # translation vector 
        self.source_pcd.transform(np.linalg.inv(tr))
        self.source_pcd_raw.transform(np.linalg.inv(tr))
        self.gt = tr
        
    def demo_transformation_geometry(self):
        '''Demonstrates geometry-based registration with custom transformation.

        This method prepares the point clouds for geometry-based registration.
        It scales both the source and target point clouds, downsamples them, applies a custom transformation matrix to the source,
        and updates the ground truth transformation matrix accordingly.

        Returns:
            None
        '''
        self.source_pcd.scale(0.0025, center= (0, 0, 0))
        self.target_pcd.scale(0.0025, center=(0, 0, 0))
        self.source_pcd = self.source_pcd.voxel_down_sample(voxel_size=0.05)
        self.target_pcd = self.target_pcd.voxel_down_sample(voxel_size=0.05)
        th = np.deg2rad(30.0)
        tr =np.array([[np.cos(th), -np.sin(th), 0.0, 250.0],
                            [np.sin(th), np.cos(th), 0.0, 65.0],
                            [0, 0, 1.0, 30.0],
                            [0.0, 0.0, 0.0, 1.0]])
        tr[:3, 3] *= 0.0025 # translation vector 
        self.source_pcd.transform(np.linalg.inv(tr))
        self.source_pcd_raw.transform(np.linalg.inv(tr))
        self.gt = tr

if  __name__ == '__main__':
    #demo test showcase
    t = Register("demo/0.ply", "demo/1.ply", gt=None, image_source="demo/0.png", image_target="demo/1.png")
    print("GEOMETRIC TEST")
    t.demo_transformation_geometry()
    for geo_method in [t.cpd_register(), t.geotransformer_register(), t.teaser_register("FPFH"), t.gcnet_register(), t.mac_register("FPFH"), t.pointdsc_register("FPFH")]:
        geo_method
    print("SUPERGLUE TEST")
    t1 = Register("demo/0.ply", "demo/1.ply", gt=None, image_source="demo/0.png", image_target="demo/1.png")
    t2 = Register("demo/0.ply", "demo/1.ply", gt=None, image_source="demo/0.png", image_target="demo/1.png")
    t3 = Register("demo/0.ply", "demo/1.ply", gt=None, image_source="demo/0.png", image_target="demo/1.png")
    t1.demo_transformation_texture('Superglue')
    t2.demo_transformation_texture('Superglue')
    t3.demo_transformation_texture('Superglue')
    
    for superglue_method in [t1.teaser_register("Superglue"), t2.mac_register("Superglue"), t3.pointdsc_register("Superglue")]:
        superglue_method
    print("PDC-NET+ TEST")
    t1 = Register("demo/0.ply", "demo/1.ply", gt=None, image_source="demo/0.png", image_target="demo/1.png")
    t2 = Register("demo/0.ply", "demo/1.ply", gt=None, image_source="demo/0.png", image_target="demo/1.png")
    t3 = Register("demo/0.ply", "demo/1.ply", gt=None, image_source="demo/0.png", image_target="demo/1.png")
    t1.demo_transformation_texture('PDC-Net+')
    t2.demo_transformation_texture('PDC-Net+')
    t3.demo_transformation_texture('PDC-Net+')
    for pdcnet_method in [t1.teaser_register("PDC-Net+"), t2.mac_register("PDC-Net+"), t3.pointdsc_register("PDC-Net+")]:
        pdcnet_method
    