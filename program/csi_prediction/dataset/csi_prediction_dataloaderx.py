import torch
from prefetch_generator import BackgroundGenerator

class my_dataloaderx(torch.utils.data.DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())