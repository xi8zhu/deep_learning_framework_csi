import torch
from tqdm import tqdm
from torch import nn
from program.csi_prediction.tools.cal_sgcs import fun_cal_sgcs
from thop import profile
import time


class convlstm():
    def __init__(self, dataloader, model, optimizer, recorder, gpu_id, total_cfg):
        self.train_dataloader, self.test_dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)
        self.total_cfg = total_cfg
        self.l2_lambda = total_cfg.module.ConvLSTM.l2_lambda
        self.mse_lambda = total_cfg.module.ConvLSTM.mse_lambda
        self.sgcs_lambda = total_cfg.module.ConvLSTM.sgcs_lambda

    def train(self, start_epoch=0, epochs=1):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            for idx, (inputs, targets) in tqdm(enumerate(self.train_dataloader)): 

                batch_size_now,_,_,_,_,_ = inputs.shape

                data_x = inputs.reshape(batch_size_now, 4, 12, 32, 4)
                data_y = targets.reshape(batch_size_now, 12, 32, 4)

                x = data_x.permute(0, 1, 4, 2, 3)
                y = data_y.permute(0, 3, 1, 2)

                x = x.to(device=self.device)
                y = y.to(device=self.device)
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
                    'iter' : idx + epoch * len(self.train_dataloader),
                    'model': self.model, 
                    'loss': loss,
                    'loss_sgcs':(1 - loss_sgcs), 
                    'loss_mse':loss_mse, 
                    'loss_l2_norm':loss_l2_norm, 
                    'cfg': self.total_cfg,
                    'epoch':epoch
                }
                self.recorder.train_log(log)
    
    def test(self):
        self.model.eval()
        with torch.no_grad():
            for idx, (inputs, targets) in tqdm(enumerate(self.test_dataloader)): 

                batch_size_now, _, _, _, _, _ = inputs.shape

                data_x = inputs.reshape(batch_size_now, 4, 12, 32, 4)
                data_y = targets.reshape(batch_size_now, 12, 32, 4)

                x = data_x.permute(0, 1, 4, 2, 3)
                y = data_y.permute(0, 3, 1, 2)

                x = x.to(device=self.device)
                y = y.to(device=self.device)
                y_pred = self.model(x)

                fun_mse = nn.MSELoss()

                l2_lambda = self.l2_lambda
                mse_lambda = self.mse_lambda
                sgcs_lambda = self.sgcs_lambda

                loss_l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters()) 
                loss_mse = fun_mse(y, y_pred)
                loss_sgcs = fun_cal_sgcs(y, y_pred)

                loss = mse_lambda * (1 - loss_sgcs) + sgcs_lambda * loss_mse + l2_lambda * loss_l2_norm
            print(f"loss_l2_norm: {loss_l2_norm}")
            print(f"loss_mse: {loss_mse}")
            print(f"loss_sgcs: {loss_sgcs}")

            model = self.model
            num_params = 0
            for p in self.model.parameters(): 
                num_params += p.numel() # numel()获取tensor中一共包含多少个元素
                # print(p.numel())
            print(f"The total number of params is {num_params}")

            for device in [self.device, 'cpu']:
                print(device)
                input = torch.randn(1, 4, 4, 12, 32)
                input = input.to(device=device)
                model = model.to(device=device)
                Flops, params = profile(self.model, inputs=(input,)) # macs
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

                print(f"Average time per inference ({device}): {avg_time_per_inference:.6f} seconds")
                print(f"Inferences per second ({device}): {1.0 / avg_time_per_inference:.2f} IPS")

            log = {
                    'model': self.model, 
                    'loss': loss,
                    'loss_sgcs':(1 - loss_sgcs), 
                    'loss_mse':loss_mse, 
                    'loss_l2_norm':loss_l2_norm, 
                    'cfg': self.total_cfg,
                }
            self.recorder.test_log(log)