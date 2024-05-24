import open3d as o3d
import numpy as np
import preprocessing
from error_calculation import calculate_error
import copy
from models import Models


class Register():
    def __init__(self, source_pcd, target_pcd, gt = None):
        self.source_pcd_raw = o3d.io.read_point_cloud(source_pcd)
        self.target_pcd_raw = o3d.io.read_point_cloud(target_pcd)
        self.source_pcd = copy.deepcopy(self.source_pcd_raw)
        self.target_pcd = copy.deepcopy(self.target_pcd_raw)
        self.gt = np.load(gt)
        self.est = None
        self.copy_of_source = copy.deepcopy(self.source_pcd_raw)


    def cpd_register(self, cuda = False):
        self.est, total_time = Models.cpd(copy.deepcopy(self.source_pcd), copy.deepcopy(self.target_pcd), cuda)
        print(f"CPD registration done! Total time: {total_time} sec.")
        self.source_pcd_raw.transform(self.est)
        self._calculate_error()
        print(f"Rotation error:{self.deg_e}, translation error: {self.tran_e}")


    def _calculate_error(self):
        if self.est is not None and self.gt is not None:
            self.deg_e, self.tran_e = calculate_error(self.gt, self.est)
    
    def icp_refinement(self, threshold):
        if self.est is not None:
            icp_reg = o3d.pipelines.registration.registration_icp(
            self.source_pcd, self.target_pcd, threshold, self.est,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)) 
            if self.gt is not None:
                temp_deg_e, temp_tran_e = calculate_error(self.gt, icp_reg)
                print(f"Without ICP errors: rot {self.deg_e}, tran {self.tran_e} \n With ICP errors: rot {temp_deg_e}, tran {temp_tran_e}")                 
            icp_pcd = copy.deepcopy(self.source_pcd_raw).transform(icp_reg)
            return icp_pcd
        
    def visualize_result(self):
        o3d.visualization.draw_geometries([self.source_pcd_raw, self.target_pcd_raw])

    def compare(self):
        self.copy_of_source.paint_uniform_color([1, 0, 0])
        self.target_pcd_raw.paint_uniform_color([0, 1, 0])
        self.source_pcd_raw.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([self.copy_of_source, self.target_pcd_raw])
        o3d.visualization.draw_geometries([self.source_pcd_raw, self.target_pcd_raw])

    
    def get_pcd(self):
        return self.source_pcd_raw, self.target_pcd_raw



t = Register("0.pcd", "1.pcd", "gt_0.npy")
t.cpd_register()
t.visualize_result()