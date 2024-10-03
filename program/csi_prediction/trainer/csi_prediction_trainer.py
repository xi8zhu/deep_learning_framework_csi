import torch
from tqdm import tqdm
from torch import nn




class convlstm_trainer():
    def __init__(self, dataloader, model, optimizer, recorder, gpu_id, total_cfg):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)
        self.total_cfg = total_cfg

    def train(self, start_epoch=0, epochs=1):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            for idx, (inputs, targets) in tqdm(enumerate(self.dataloader)):
                loss_function = nn.MSELoss()

                batch_size_now,_,_,_,_,_ = inputs.shape

                data_x = inputs.reshape(batch_size_now, 4, 12, 32, 4)
                data_y = targets.reshape(batch_size_now, 12, 32, 4)

                x = data_x.permute(0,1,4,2,3)
                y = data_y.permute(0,3,1,2)

                x = x.to(device=self.device)
                y = y.to(device=self.device)
                y_pred = self.model(x)
                l2_lambda = 0.0001
                l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters()) 
                loss = 1 - loss_function(y_pred, y) + l2_lambda * l2_norm
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                log = {
                    'iter' : idx + epoch * len(self.dataloader),
                    'model': self.model, 
                    'loss': loss,
                    'cfg': self.total_cfg,
                    'epoch':epoch
                }
                self.recorder.log(log)