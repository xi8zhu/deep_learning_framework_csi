from tensorboardX import SummaryWriter
import torch
import os
import numpy as np
from datetime import datetime
import json

# checkpoint_path + name: model parameters
# result_path + name 

class convlstm_recorder():
    def __init__(self, cfg, only_test):
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
        self.result_path = cfg.result_path

        self.save_freq = cfg.save_freq
        self.only_test = only_test
        if not self.only_test:
            os.makedirs(self.checkpoint_path, exist_ok=True)
            os.makedirs('%s/%s' % (self.checkpoint_path, self.file_name), exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs('%s/%s' % (self.result_path, self.file_name), exist_ok=True)

    
    def train_log(self, log_data):
        if self.save_total_cfg:
            with open('%s/%s/config_file.yaml' % (self.checkpoint_path, self.file_name), 'w') as f:
                f.write(log_data['cfg'].dump())
            self.save_total_cfg = False
        self.logger.add_scalar('training_loss', log_data['loss'], log_data['iter'])
        self.logger.add_scalar('training_sgcs', log_data['sgcs'], log_data['iter'])
        self.logger.add_scalar('training_loss_sgcs', log_data['loss_sgcs'], log_data['iter'])
        self.logger.add_scalar('training_loss_mse', log_data['loss_mse'], log_data['iter'])
        self.logger.add_scalar('training_loss_l2_norm', log_data['loss_l2_norm'], log_data['iter'])

        if log_data['iter'] % self.save_freq == 0 or log_data['last_train']:
            print('saving checkpoint.')
            torch.save(log_data['model'].state_dict(), '%s/%s_ConvLSTM_latest_%s' % (self.checkpoint_path, self.name, self.current_time))
            torch.save(log_data['model'].state_dict(), '%s/%s/ConvLSTM_epoch_%d' % (self.checkpoint_path, self.file_name, log_data['epoch']))

    def overfitting_log(self, log_data):
        self.logger.add_scalars('overfitting_loss', 
            {'train_loss': log_data['train_loss'], 'test_loss': log_data['test_loss']}, log_data['epoch'])
        
        self.logger.add_scalars('overfitting_sgcs', 
            {'train_sgcs': log_data['train_sgcs'], 'test_sgcs': log_data['test_sgcs']}, log_data['epoch'])

        self.logger.add_scalars('overfitting_loss_mse',
            {'train_mse_loss': log_data['train_loss_mse'], 'test_mse_loss': log_data['test_loss_mse']}, log_data['epoch'])
        
        self.logger.add_scalars('overfitting_loss_sgcs', 
            {'train_loss_sgcs': log_data['train_loss_sgcs'], 'test_loss_sgcs': log_data['test_loss_sgcs']}, log_data['epoch'])



    def result_log(self, log_data):
        with open('%s/%s/config_file.yaml' % (self.result_path, self.file_name), 'w') as f:
            f.write(log_data['cfg'].dump())
        log_data.pop('cfg')
        with open('%s/%s/model_summary.txt' % (self.result_path, self.file_name), 'w') as f:
            print(log_data['model'], file=f)
        log_data.pop('model')
        with open('%s/%s/result.txt' % (self.result_path, self.file_name), 'w') as json_file:
            json.dump(log_data, json_file, indent=4)
        