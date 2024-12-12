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
        self.cfg.only_test = False

        self.cfg.lr_net = 0.0                                   # learning rate for models and networks
        
        self.cfg.dataset = CN()
        self.cfg.dataset.batch_size = 1                                 # recommend batch_size = 1
        self.cfg.dataset.total_data = False
        self.cfg.dataset.data_split_flag = True
        self.cfg.dataset.data_split = ''
        self.cfg.dataset.dataroot = ''                          # root of the dataset
        self.cfg.dataset.dataroot_x = '' 
        self.cfg.dataset.dataroot_y = '' 
        
        self.cfg.module = CN()
        self.cfg.module.loss = CN()
        self.cfg.module.loss.l2_lambda = 0.01
        self.cfg.module.loss.mse_lambda = 0.01
        self.cfg.module.loss.sgcs_lambda = 0.01

        self.cfg.module.model_name = ''
        self.cfg.module.ConvLSTM = CN()
        self.cfg.module.ConvLSTM.convlstm_layer = 2
        self.cfg.module.ConvLSTM.CNN_layer = 3
        self.cfg.module.ConvLSTM.convlstm_input_dim = [4, 64]
        self.cfg.module.ConvLSTM.convlstm_output_dim = [64, 128]
        self.cfg.module.ConvLSTM.convlstm_kernel_size = [3, 5]
        self.cfg.module.ConvLSTM.CNN_input_dim = [128, 128, 64]
        self.cfg.module.ConvLSTM.CNN_output_dim = [128, 64, 64]
        self.cfg.module.ConvLSTM.CNN_kernel_size = [3, 3, 3]
        self.cfg.module.ConvLSTM.CNN_stride = [2, 2, 1]

        self.cfg.module.TransLSTM = CN()
        self.cfg.module.TransLSTM.layers_num = 6
        self.cfg.module.TransLSTM.d_model = 512
        self.cfg.module.TransLSTM.d_ff = 2048
        self.cfg.module.TransLSTM.h = 8
        self.cfg.module.TransLSTM.dropout = 0.1

        self.cfg.recorder = CN()
        self.cfg.recorder.name = ''                             # name of the avatar
        self.cfg.recorder.logdir = ''                           # directory of the tensorboard log
        self.cfg.recorder.checkpoint_path = ''                  # path to the saved checkpoints
        self.cfg.recorder.result_path = ''
        self.cfg.recorder.save_freq = 1                         # how often the checkpoints are saved
        self.cfg.recorder.comment = '' 
        self.cfg.recorder.save_total_cfg = False
    

