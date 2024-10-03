from tensorboardX import SummaryWriter
import torch
import os
import numpy as np
# checkpoint_path + name: model parameters
# result_path + name 

class convlstm_recorder():
    def __init__(self, cfg):
        self.logdir = cfg.logdir
        self.logger = SummaryWriter(self.logdir)

        self.name = cfg.name
        self.checkpoint_path = cfg.checkpoint_path
        
        self.save_freq = cfg.save_freq
        self.show_freq = cfg.show_freq

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs('%s/%s' % (self.checkpoint_path, self.name), exist_ok=True)

    
    def log(self, log_data):
        self.logger.add_scalar('loss', log_data['loss'], log_data['iter'])

        if log_data['iter'] % self.save_freq == 0:
            print('saving checkpoint.')
            torch.save(log_data['model'].state_dict(), '%s/%sConvLSTM_latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['model'].state_dict(), '%s/%s/ConvLSTM_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
