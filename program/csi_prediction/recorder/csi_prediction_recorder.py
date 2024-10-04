from tensorboardX import SummaryWriter
import torch
import os
import numpy as np
from datetime import datetime
# checkpoint_path + name: model parameters
# result_path + name 

class convlstm_recorder():
    def __init__(self, cfg):
        self.save_total_cfg = cfg.save_total_cfg

        self.logdir = cfg.logdir
        self.comment = cfg.comment
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # logdir = os.path.join(
        #     self.logdir, self.current_time + '_' + self.comment)
        logdir = os.path.join(
            self.logdir, self.current_time)
        self.logger = SummaryWriter(logdir)

        self.name = cfg.name
        self.file_name = os.path.join(
            self.name + '_' + self.current_time + '_' + self.comment)
        self.checkpoint_path = cfg.checkpoint_path
        
        self.save_freq = cfg.save_freq

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs('%s/%s' % (self.checkpoint_path, self.file_name), exist_ok=True)

    
    def log(self, log_data):
        if self.save_total_cfg:
            with open('%s/%s/config_file.yaml' % (self.checkpoint_path, self.file_name), 'w') as f:
                f.write(log_data['cfg'].dump())
            self.save_total_cfg = False
        self.logger.add_scalar('loss', log_data['loss'], log_data['iter'])
        self.logger.add_scalar('loss_sgcs', log_data['loss_sgcs'], log_data['iter'])
        self.logger.add_scalar('loss_mse', log_data['loss_mse'], log_data['iter'])
        self.logger.add_scalar('loss_l2_norm', log_data['loss_l2_norm'], log_data['iter'])

        if log_data['iter'] % self.save_freq == 0:
            print('saving checkpoint.')
            torch.save(log_data['model'].state_dict(), '%s/%s_ConvLSTM_latest_%s' % (self.checkpoint_path, self.name, self.current_time))
            torch.save(log_data['model'].state_dict(), '%s/%s/ConvLSTM_epoch_%d' % (self.checkpoint_path, self.file_name, log_data['epoch']))
