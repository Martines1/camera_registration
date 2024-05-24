import numpy as np
from models_dir import cpd_run

class Models():
    def cpd(source, target, cuda_):
        return cpd_run.register(source, target, cuda_)
    
