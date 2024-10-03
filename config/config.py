import os
from yacs.config import CfgNode as CN
 

class config_base():

    def __init__(self):
        self.cfg = CN()
    
    def get_cfg(self):
        return  self.cfg.clone()
    
    def load(self,config_file):
         self.cfg.defrost()
         self.cfg.merge_from_file(config_file)
         self.cfg.freeze()

#In the YAML file you need to merge, there cannot be any parameters that do not exist 
# in the default parameter, otherwise an error will be reported. 
# However, there can be fewer parameters than those set in the default file.
# For example, the name parameter is in the default file, 
# which does not require specific changes. You can choose not to set the name key in YAML.

class config_train(config_base):

    def __init__(self):
        super(config_train, self).__init__()
        self.cfg.gpu_id = 0                                     # which gpu is used
        self.cfg.load_csi_prediction_checkpoint = ''            # checkpoint path of csi prediction 
        self.cfg.train_epoch = 0

        self.cfg.load_meshhead_checkpoint = ''                  # checkpoint path of mesh head
        self.cfg.load_gaussianhead_checkpoint = ''              # checkpoint path of gaussian head
        self.cfg.load_supres_checkpoint = ''                    # checkpoint path of super resolution network
        self.cfg.load_delta_poses_checkpoint = ''               # checkpoint path of per-frame offset of head pose
        self.cfg.lr_net = 0.0                                   # learning rate for models and networks
        self.cfg.batch_size = 1                                 # recommend batch_size = 1
        self.cfg.optimize_pose = False                          # optimize delta_poses or not
        
        self.cfg.dataset = CN()
        self.cfg.dataset.total_data = False
        self.cfg.dataset.data_split_flag = True
        self.cfg.dataset.data_split = ''
        self.cfg.dataset.dataroot = ''                          # root of the dataset
        self.cfg.dataset.dataroot_x = '' 
        self.cfg.dataset.dataroot_y = '' 
        
        self.cfg.module = CN()
        self.cfg.module.name = ''

        self.cfg.meshheadmodule = CN()
        self.cfg.meshheadmodule.geo_mlp = []                    # dimensions of geometry MLP
        self.cfg.meshheadmodule.exp_color_mlp = []              # dimensions of expression color MLP
        self.cfg.meshheadmodule.pose_color_mlp = []             # dimensions of pose color MLP
        self.cfg.meshheadmodule.exp_deform_mlp = []             # dimensions of expression deformation MLP
        self.cfg.meshheadmodule.pose_deform_mlp = []            # dimensions of pose deformation MLP
        self.cfg.meshheadmodule.pos_freq = 4                    # frequency of positional encoding
        self.cfg.meshheadmodule.model_bbox = []                 # bounding box of the head model
        self.cfg.meshheadmodule.dist_threshold_near = 0.1       # threshold t1
        self.cfg.meshheadmodule.dist_threshold_far = 0.2        # thresgold t2
        self.cfg.meshheadmodule.deform_scale = 0.3              # scale factor for deformation
        self.cfg.meshheadmodule.subdivide = False               # subdivide the tetmesh (resolution: 128 --> 256) or not

        self.cfg.supresmodule = CN()
        self.cfg.supresmodule.input_dim = 32                    # input dim, equal to the channel number of the multi-channel color
        self.cfg.supresmodule.output_dim = 3                    # output dim, euqal to the channel number of the final image
        self.cfg.supresmodule.network_capacity = 64             # dimension of the network's last conv layer

        self.cfg.gaussianheadmodule = CN()
        self.cfg.gaussianheadmodule.num_add_mouth_points = 0    # number of the points added around mouth landmarks while initialization
        self.cfg.gaussianheadmodule.exp_color_mlp = []          # dimensions of expression color MLP
        self.cfg.gaussianheadmodule.pose_color_mlp = []         # dimensions of pose color MLP
        self.cfg.gaussianheadmodule.exp_attributes_mlp = []     # dimensions of expression attribute MLP
        self.cfg.gaussianheadmodule.pose_attributes_mlp = []    # dimensions of pose attribute MLP
        self.cfg.gaussianheadmodule.exp_deform_mlp = []         # dimensions of expression deformation MLP
        self.cfg.gaussianheadmodule.pose_deform_mlp = []        # dimensions of pose deformation MLP
        self.cfg.gaussianheadmodule.exp_coeffs_dim = 64         # dimension of the expression coefficients
        self.cfg.gaussianheadmodule.pos_freq = 4                # frequency of positional encoding
        self.cfg.gaussianheadmodule.dist_threshold_near = 0.1   # threshold t1
        self.cfg.gaussianheadmodule.dist_threshold_far = 0.2    # thresgold t2
        self.cfg.gaussianheadmodule.deform_scale = 0.3          # scale factor for deformation
        self.cfg.gaussianheadmodule.attributes_scale = 0.05     # scale factor for attribute offset

        self.cfg.recorder = CN()
        self.cfg.recorder.name = ''                             # name of the avatar
        self.cfg.recorder.logdir = ''                           # directory of the tensorboard log
        self.cfg.recorder.checkpoint_path = ''                  # path to the saved checkpoints
        self.cfg.recorder.result_path = ''                      # path to the visualization results
        self.cfg.recorder.save_freq = 1                         # how often the checkpoints are saved
        self.cfg.recorder.show_freq = 1                         # how often the visualization results are saved


