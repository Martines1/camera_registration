class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir
        self.pre_trained_models_dir = ''
        self.megadepth = ''
        self.megadepth_csv = ''
        self.robotcar = ''
        self.robotcar_csv = ''
        self.hp = ''
        self.eth3d = ''
        self.kitti2012 = ''
        self.kitti2015 = ''
        self.sintel = ''
        self.scannet_test = ''
        self.yfcc = ''
        self.tss = ''
        self.PFPascal = ''
        self.PFWillow = ''
        self.spair = ''
        self.caltech = ''
        self.training_cad_520 = ''
        self.validation_cad_520 = ''
        self.coco = ''
        self.megadepth_training = ''
