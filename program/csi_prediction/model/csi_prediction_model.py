import torch
from tqdm import tqdm
from torch import nn
from program.csi_prediction.tools.cal_sgcs import fun_cal_sgcs
from program.csi_prediction.dataset.csi_prediction_dataloaderx import my_dataloaderx
from program.csi_prediction.tools.data_split_validate import data_split_validate
from torch.utils.data import random_split
from thop import profile
import time


class convlstm():
    def __init__(self, dataset, model, optimizer, recorder, gpu_id, total_cfg):
        self.dataset = dataset
        self.data_split_flag = total_cfg.dataset.data_split_flag
        self.batch_size = total_cfg.dataset.batch_size
        if self.data_split_flag:
            self.dataloader, self.test_dataloader = self.__load_split_dataset(dataset, total_cfg.dataset)
        else:
            self.dataloader = my_dataloaderx(dataset, batch_size = self.batch_size, shuffle=True)
        self.model = model
        self.optimizer = optimizer
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)
        self.total_cfg = total_cfg
        self.l2_lambda = total_cfg.module.ConvLSTM.l2_lambda
        self.mse_lambda = total_cfg.module.ConvLSTM.mse_lambda
        self.sgcs_lambda = total_cfg.module.ConvLSTM.sgcs_lambda

    def __load_split_dataset(self, dataset, cfg):
        split_array = data_split_validate(cfg.data_split)
        train_dataset, test_dataset, _ = random_split(dataset, split_array)
        # validation dataset is temporily not used! And we can ignore the warnings.
        train_dataloader = my_dataloaderx(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        test_dataloader = my_dataloaderx(test_dataset, batch_size=cfg.batch_size, shuffle=True)
        return train_dataloader, test_dataloader


    def __process_data_for_model(self, inputs, targets, device):
        batch_size_now = inputs.shape[0]

        data_x = inputs.reshape(batch_size_now, 4, 12, 32, 4)
        data_y = targets.reshape(batch_size_now, 12, 32, 4)

        x = data_x.permute(0, 1, 4, 2, 3)
        y = data_y.permute(0, 3, 1, 2)

        x = x.to(device = device)
        y = y.to(device = device)

        return x, y

    def train(self, start_epoch=0, epochs=1):
        for epoch in range(start_epoch, epochs):
            self.model.train()
            sum_loss = {
                "loss" : 0,
                "loss_mse" : 0,
                "loss_sgcs" : 0
            }
            for idx, (inputs, targets) in tqdm(enumerate(self.dataloader)): 
                batch_size = inputs.shape[0]
                x, y = self.__process_data_for_model(inputs, targets, self.device)
                y_pred = self.model(x)

                l2_lambda = self.l2_lambda
                mse_lambda = self.mse_lambda
                sgcs_lambda = self.sgcs_lambda

                loss_l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters()) 
                fun_mse = nn.MSELoss()
                loss_mse = fun_mse(y, y_pred)
                loss_sgcs = fun_cal_sgcs(y, y_pred)

                loss = mse_lambda * (1 - loss_sgcs) + sgcs_lambda * loss_mse + l2_lambda * loss_l2_norm
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                log = {
                    'last_train':False,
                    'iter' : idx + epoch * len(self.dataloader),
                    'model': self.model, 
                    'loss': loss,
                    'sgcs': loss_sgcs, 
                    'loss_sgcs':(1 - loss_sgcs), 
                    'loss_mse':loss_mse, 
                    'loss_l2_norm':loss_l2_norm, 
                    'cfg': self.total_cfg,
                    'epoch':epoch
                }
                self.recorder.train_log(log)
                
                sum_loss['loss'] = sum_loss['loss'] + loss
                sum_loss['loss_mse'] = sum_loss['loss_mse'] + loss_mse
                sum_loss['loss_sgcs'] = sum_loss['loss_sgcs'] + loss_sgcs

            if self.data_split_flag:
                train_length = len(self.dataloader)
                # 必须划分过的数据集，可被batch size整除
                test_log = {
                    'epoch': epoch,
                    'train_loss': sum_loss['loss']/train_length,
                    'train_sgcs': sum_loss['loss_sgcs']/train_length,
                    'train_loss_sgcs':(1 - sum_loss['loss_sgcs']/train_length), 
                    'train_loss_mse':sum_loss['loss_mse']/train_length,
                }
                loss = self.__model_test(self.test_dataloader)
                test_log['test_loss'] = loss['test_loss']
                test_log['test_loss_sgcs'] =loss['test_loss_sgcs']
                test_log['test_loss_mse'] = loss['test_loss_mse']
                test_log['test_sgcs'] = loss['test_sgcs']

                self.recorder.overfitting_log(test_log)

        log['last_trarin'] = True
        self.recorder.train_log(log)
        self.last_test(self.test_dataloader)
                
    def __model_test(self, test_dataloader):
        self.model.eval()
        with torch.no_grad():
            sum_loss = {
                "loss" : 0,
                "loss_mse" : 0,
                "loss_sgcs" : 0
            }
            test_length = len(test_dataloader)
            for idx, (inputs, targets) in tqdm(enumerate(test_dataloader)): 
                batch_size = inputs.shape[0]
                x, y = self.__process_data_for_model(inputs, targets, self.device)
                y_pred = self.model(x)

                fun_mse = nn.MSELoss()

                l2_lambda = self.l2_lambda
                mse_lambda = self.mse_lambda
                sgcs_lambda = self.sgcs_lambda

                loss_l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters()) 
                loss_mse = fun_mse(y, y_pred)
                loss_sgcs = fun_cal_sgcs(y, y_pred)

                loss = mse_lambda * (1 - loss_sgcs) + sgcs_lambda * loss_mse + l2_lambda * loss_l2_norm

                sum_loss['loss'] = sum_loss['loss'] + loss
                sum_loss['loss_mse'] = sum_loss['loss_mse'] + loss_mse
                sum_loss['loss_sgcs'] = sum_loss['loss_sgcs'] + loss_sgcs
            
            return {
                'test_loss': sum_loss['loss']/test_length, 
                'test_loss_mse': sum_loss['loss_mse']/test_length, 
                'test_sgcs': sum_loss['loss_sgcs']/test_length,
                'test_loss_sgcs': 1 - sum_loss['loss_sgcs']/test_length,
                'loss_l2_norm': loss_l2_norm,
            }

    def last_test(self, test_dataloader):
            loss = self.__model_test(test_dataloader)
            model = self.model
            num_params = 0
            for p in model.parameters(): 
                num_params += p.numel() # numel()获取tensor中一共包含多少个元素
                # print(p.numel())
            print(f"The total number of params is {num_params}")
            infer_time = []
            for device in [self.device, 'cpu']:
                print(device)
                input = torch.randn(1, 4, 4, 12, 32)
                input = input.to(device=device)
                model = model.to(device=device)
                Flops, params = profile(model, inputs=(input,)) # macs
                print('FLOPs: % .4fM'%(2 * Flops / 1000000))# 计算量
                print('params参数量: % .4fM'% (params / 1000000)) 
                model(input)
                if device == self.device:
                    torch.cuda.synchronize()  # GPU 同步
                start_time = time.time()
                # 进行推理，重复多次来计算平均时间
                num_inferences = 100
                with torch.no_grad():
                    for _ in range(num_inferences):
                        if device == self.device:
                            torch.cuda.synchronize()  # GPU 同步
                        model(input)

                # 结束时间测量
                end_time = time.time()

                # 计算推理时间
                total_time = end_time - start_time
                avg_time_per_inference = total_time / num_inferences
                infer_time.append(avg_time_per_inference)
                print(f"Average time per inference ({device}): {avg_time_per_inference:.6f} seconds")
                print(f"Inferences per second ({device}): {1.0 / avg_time_per_inference:.2f} IPS")

            log = {
                    'model': model.to(self.device), 
                    'loss': str(loss['test_loss']),
                    'sgcs':str(1 - loss['test_loss_sgcs']), 
                    'loss_mse':str(loss['test_loss_mse']), 
                    'loss_l2_norm':str(loss['loss_l2_norm']), 
                    'cfg': self.total_cfg,
                    'gpu_infer_time': f"{infer_time[0]:.6f}s",
                    'cpu_infer_time': f"{infer_time[1]:.6f}s",
                    'FLOPs': "% .4fM"%(2 * Flops / 1000000),
                    'params': " % .4fM"% (params / 1000000)
                }
            self.recorder.result_log(log)