import sys
sys.path.append('./program/csi_prediction/module/')
sys.path.append('./program/csi_prediction/model/')

# print(sys.path)
import os
import torch
import argparse
from config.config import config_train
from program.csi_prediction.dataset.csi_prediction_dataset import my_dataset
from program.csi_prediction.module.csi_prediction_module import choose_module
from program.csi_prediction.recorder.csi_prediction_recorder import convlstm_recorder as Recorder
from program.csi_prediction.model.csi_prediction_model import convlstm as Model


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

    device = torch.device('cuda:%d' % cfg.gpu_id)
    print(device)
    torch.cuda.set_device(cfg.gpu_id)
    # ToDo :choose_module添加返回训练器
    model = choose_module(cfg.module).to(device)
    print(model)
    # 目前没用到这个
    if os.path.exists(cfg.load_csi_prediction_checkpoint):
        model.load_state_dict(torch.load(cfg.load_csi_prediction_checkpoint, map_location=lambda storage, loc: storage))
    # else:
    #     model.pre_train(300, device)
    recorder = Recorder(cfg.recorder, cfg.only_test)

    optimizer = torch.optim.Adam([{'params':model.parameters(), 'lr' : cfg.lr_net}])
    all_model = Model(dataset, model, optimizer, recorder, cfg.gpu_id, cfg)
    all_model.train(0, cfg.train_epoch)
    # 设计时，不在train方法里加config，原因是希望经过不同模型选择后实例化的trainer拥有共性参数...这样写估计以后听不懂
    if cfg.only_test:
        all_model.last_test()