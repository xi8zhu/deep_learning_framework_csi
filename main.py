import sys
sys.path.append('./program/csi_prediction/module/')
# print(sys.path)

import os
import torch
import argparse
from config.config import config_train
from program.csi_prediction.dataset.csi_prediction_dataset import my_dataset
from program.csi_prediction.dataset.csi_prediction_dataloaderx import my_dataloaderx
from program.csi_prediction.tools.data_split_validate import data_split_validate
from program.csi_prediction.module.csi_prediction_module import choose_module
from program.csi_prediction.recorder.csi_prediction_recorder import convlstm_recorder as Recorder
from program.csi_prediction.trainer.csi_prediction_trainer import convlstm_trainer as Trainer
from torch.utils.data import random_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/csi_prediction/train.yaml')
    arg = parser.parse_args()

    cfg = config_train()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()
    # print(cfg)

    # data load
    dataset = my_dataset(cfg.dataset)
    print(dataset.__len__())
    print(dataset.__getitem__(1)[0].shape)
    if cfg.dataset.data_split_flag:
        split_array = data_split_validate(cfg.dataset.data_split)
        train_dataset, test_dataset, _ = random_split(dataset, split_array)
        # validation dataset is temporily not used! And we can ignore the warnings.
        train_dataloader = my_dataloaderx(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        test_dataloader = my_dataloaderx(test_dataset, batch_size=cfg.batch_size, shuffle=True)
    # print(train_dataloader)
    # print(test_dataloader)
    device = torch.device('cuda:%d' % cfg.gpu_id)
    torch.cuda.set_device(cfg.gpu_id)

    model = choose_module(cfg.module).to(device)
    print(model)
    if os.path.exists(cfg.load_csi_prediction_checkpoint):
        model.load_state_dict(torch.load(cfg.load_csi_prediction_checkpoint, map_location=lambda storage, loc: storage))
    # else:
    #     model.pre_train(300, device)
    recorder = Recorder(cfg.recorder)

    optimizer = torch.optim.Adam([{'params':model.parameters(), 'lr' : cfg.lr_net}])
    trainer = Trainer(train_dataloader, model, optimizer, recorder, cfg.gpu_id, cfg)
    trainer.train(0, cfg.train_epoch)
    # tester = Tester(test_dataloader, model, optimizer, recorder, cfg.gpu_id, cfg)
    # tester.test(0, 50)