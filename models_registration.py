import numpy as np
from models_dir.coherent_point_drift import cpd_run
from models_dir.GeoTransformer import geotransformer_run
from models_dir.TEASER import teaser_run
from models_dir.MAC import mac_run
from models_dir.PointDSC import pointdsc_run
import subprocess
import open3d as o3d

class Models():
    def cpd(source, target, cuda_):
        return cpd_run.register(source, target, cuda_)
    
    def geotransformer(source, target):
        return geotransformer_run.main(source, target)
    
    def teaser(source, target, voxel_size, corrs_A, corrs_B):
        return teaser_run.register(source, target, voxel_size, corrs_A, corrs_B)

    def gcnet(source, target, voxel_size=0.01):
        o3d.io.write_point_cloud("models_dir/GCNet/source.pcd", source)
        o3d.io.write_point_cloud("models_dir/GCNet/target.pcd", target)
        command = f"conda run -n gcnet python models_dir/GCNet/gcnet_run.py --checkpoint models_dir/GCNet/weights/3dmatch.pth  --voxel_size {voxel_size} --npts 20000"
        proc = subprocess.run(command, shell=True, capture_output=True)
        t = float(proc.stdout.split()[-1])
        
        est = np.load("gcnet_t.npy")
        return est, t
    
    def mac(source, target, feat_A, feat_B):
        return mac_run.register(source, target, feat_A, feat_B)
    
    def pointdsc(source, target, feat_A, feat_B):
        return pointdsc_run.register(source, target, feat_A, feat_B)