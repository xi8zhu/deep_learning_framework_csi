import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import h5py



def fun_cal_sgcs(y_true, y_pred):
    # (batch_size, channel, subcarrier, transmit_antenna)
    batch_size, channel, subcarrier, transmit_antenna = y_true.shape
    W_true_tempo = y_true.permute(0, 2, 3, 1)
    W_pred_tempo = y_pred.permute(0, 2, 3, 1)

    W_true = W_true_tempo.reshape(batch_size, subcarrier, transmit_antenna, 2, 2)
    W_true = W_true.reshape(batch_size, -1, 2)

    W_pred = W_pred_tempo.reshape(batch_size, subcarrier, transmit_antenna, 2, 2)
    W_pred = W_pred.reshape(batch_size, -1, 2)

    W_true_re, W_true_im = W_true[..., 0], W_true[..., 1]
    W_pre_re, W_pre_im = W_pred[..., 0], W_pred[..., 1]

    numerator_re = torch.sum((W_true_re * W_pre_re + W_true_im * W_pre_im), -1)
    numerator_im = torch.sum((W_true_im * W_pre_re - W_true_re * W_pre_im), -1)
    denominator_0 = torch.sum((W_true_re * W_true_re + W_true_im * W_true_im), -1)
    denominator_1 = torch.sum((W_pre_re * W_pre_re + W_pre_im * W_pre_im), -1)
    cos_similarity = torch.sqrt(numerator_re * numerator_re + numerator_im * numerator_im) / (
                torch.sqrt(denominator_0) * torch.sqrt(denominator_1))
    cos_similarity = torch.mean(cos_similarity * cos_similarity)

    return cos_similarity


class class_cal_sgcs(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # SGCS:
        # input:(batch_size_now, frame_channel, frame_height, frame_width)
        # output:(sum cos similarity)/batch_size_now
        batch_size, frame_channel, frame_height, frame_width = input.size()
        sum = 0
        for i in range(batch_size):
            x = input[i,:,:,:]
            x = x.reshape(frame_channel*frame_height*frame_width)
            y = target[i,:,:,:]
            y = y.reshape(frame_channel*frame_height*frame_width)
            cos = torch.cosine_similarity(x, y, dim=0)
            sum += cos
        return torch.mean(sum)


if __name__ == '__main__':
    print("hello world!")
    y = h5py.File('dataset/Umi_outdoor30_21(5).mat', 'r')
    y = y['result_21']
    y = torch.from_numpy(y[...,:50]).float()
    y = y.permute(5, 4, 3, 2, 1, 0)
    y = y[:, 0, ...]
    batch_size_now, _, _, _, _ = y.shape
    y = y.reshape(batch_size_now, 12, 32, 4)
    y = y.permute(0, 3, 1, 2)
    y1 = y[..., :] - 1
    a = fun_cal_sgcs(y, y1)
    print(a)