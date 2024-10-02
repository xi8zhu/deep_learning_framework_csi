from tensorboardX import SummaryWriter
import torch
import os
import numpy as np


class convlstm_recorder():
    def __init__(self, cfg):
            self.logdir = cfg.logdir
            self.logger = SummaryWriter(self.logdir)

            self.name = cfg.name
            self.checkpoint_path = cfg.checkpoint_path
            self.result_path = cfg.result_path
            
            self.save_freq = cfg.save_freq
            self.show_freq = cfg.show_freq

            # os.makedirs(self.checkpoint_path, exist_ok=True)
            # os.makedirs(self.result_path, exist_ok=True)
            # os.makedirs('%s/%s' % (self.checkpoint_path, self.name), exist_ok=True)
            # os.makedirs('%s/%s' % (self.result_path, self.name), exist_ok=True)

class GaussianHeadTrainRecorder():
    def __init__(self, cfg):
        self.logdir = cfg.logdir
        self.logger = SummaryWriter(self.logdir)

        self.name = cfg.name
        self.checkpoint_path = cfg.checkpoint_path
        self.result_path = cfg.result_path
        
        self.save_freq = cfg.save_freq
        self.show_freq = cfg.show_freq

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs('%s/%s' % (self.checkpoint_path, self.name), exist_ok=True)
        os.makedirs('%s/%s' % (self.result_path, self.name), exist_ok=True)

    
    def log(self, log_data):
        self.logger.add_scalar('loss_rgb_hr', log_data['loss_rgb_hr'], log_data['iter'])
        self.logger.add_scalar('loss_rgb_lr', log_data['loss_rgb_lr'], log_data['iter'])
        self.logger.add_scalar('loss_vgg', log_data['loss_vgg'], log_data['iter'])

        if log_data['iter'] % self.save_freq == 0:
            print('saving checkpoint.')
            torch.save(log_data['gaussianhead'].state_dict(), '%s/%s/gaussianhead_latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['gaussianhead'].state_dict(), '%s/%s/gaussianhead_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
            torch.save(log_data['supres'].state_dict(), '%s/%s/supres_latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['supres'].state_dict(), '%s/%s/supres_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
            torch.save(log_data['delta_poses'], '%s/%s/delta_poses_latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['delta_poses'], '%s/%s/delta_poses_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))

        if log_data['iter'] % self.show_freq == 0:
            image = log_data['data']['images'][0].permute(1, 2, 0).detach().cpu().numpy()
            image = (image * 255).astype(np.uint8)[:,:,::-1]

            render_image = log_data['data']['render_images'][0, 0:3].permute(1, 2, 0).detach().cpu().numpy()
            render_image = (render_image * 255).astype(np.uint8)[:,:,::-1]

            cropped_image = log_data['data']['cropped_images'][0].permute(1, 2, 0).detach().cpu().numpy()
            cropped_image = (cropped_image * 255).astype(np.uint8)[:,:,::-1]

            supres_image = log_data['data']['supres_images'][0].permute(1, 2, 0).detach().cpu().numpy()
            supres_image = (supres_image * 255).astype(np.uint8)[:,:,::-1]

            render_image = cv2.resize(render_image, (image.shape[0], image.shape[1]))
            result = np.hstack((image, render_image, cropped_image, supres_image))
            cv2.imwrite('%s/%s/%06d.jpg' % (self.result_path, self.name, log_data['iter']), result)
