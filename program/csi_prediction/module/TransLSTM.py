import numpy as np
import argparse
import h5py
import torch
from torch import nn

from Module_Transformer import (
    EncoderDecoder, subsequent_mask, MultiHeadedAttention,
    PositionwiseFeedForward, PositionalEncoding, EncoderLayer,
    Encoder, Decoder, DecoderLayer, Generator
)
import math
import copy

class Linear1(nn.Module):
    def __init__(self, d_model, vocab):
        super(Linear1, self).__init__()
        self.lut = nn.Linear(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Linear1(d_model, src_vocab), c(position)),
        nn.Sequential(Linear1(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class default_transformer(nn.Module):
    def __init__(self, vocab = 768, N = 6, d_model = 512, d_ff = 2048, h = 8, dropout = 0.1):
        """
        N: layers
        vocab:input of linear embedding dim
        d_model: output of linear embedding dim
        d_ff:FFN hidden dim
        dropout: dropout
        """
        super(default_transformer, self).__init__()
        self.model = make_model(vocab, vocab, N, d_model, d_ff, h, dropout)
    def forward(self, src):
        ys = torch.zeros(1, 1).type_as(src)
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        output = self.model(src, src, None, tgt_mask)
        return output

class TransLSTM(nn.Module):
    def __init__(self, module_opt = None, csi_dim = 64, output_time = 1):
        super(TransLSTM, self).__init__()
        if not module_opt:
            d_model = 512
        self.transformer = default_transformer(vocab = csi_dim, d_model = d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=csi_dim, batch_first = True)
        self.output_time = output_time
        if module_opt:
            pass
    def forward(self, x):
        x1 = self.transformer(x)
        x2, (h_n, c_n) = self.lstm(x1)
        return x2[:, -self.output_time, :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/data/lzh/Umi_outdoor30_1_6_11_16(5).mat')
    arg = parser.parse_args()
    dataset_path = arg.dataset
    y = h5py.File(dataset_path, 'r')
    y = y[list(y.keys())[0]]
    y = torch.from_numpy(y[...,:50]).float()
    y = y.permute(5, 4, 3, 2, 1, 0)
    data = y[:,:,0,:,0,:]
    sample, time, tx, real_imag = data.shape
    data_train = data.reshape(sample, time, tx * real_imag)
    print(data_train.shape)
    model = TransLSTM()
    output = model(data_train)
    print(output.shape)
