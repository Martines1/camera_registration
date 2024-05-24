import copy
import numpy as np
import open3d as o3
import time
from probreg import cpd

def register(source, target, cuda_):
            start_time = time.time()
            tf_param, _, _ = cpd.registration_cpd(source, target, use_cuda=cuda_)
            end_time = time.time() - start_time
            est = np.eye(4)
            if cuda_:
                est[:3, 3] = tf_param.t.get()
                est[:3,:3] = tf_param.rot.get()
            else:
                est[:3, 3] = tf_param.t
                est[:3,:3] = tf_param.rot
            return est, end_time
            '''
            deg, tran = calculate_error(gt, est)
            icp_sol = o3.pipelines.registration.registration_icp(
            source, target, 0.05, est,
            o3.pipelines.registration.TransformationEstimationPointToPoint(),
            o3.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))
            T_icp = icp_sol.transformation
            icp_deg, icp_tran = calculate_error(gt, T_icp)
            result = copy.deepcopy(source)
            result.transform(T_icp)
            source.paint_uniform_color([1, 0, 0])
            target.paint_uniform_color([0, 1, 0])
            result.paint_uniform_color([0, 0, 1])
            #o3.visualization.draw_geometries([source, target, result])
            '''
            


