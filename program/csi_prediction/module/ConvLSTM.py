import torch.nn as nn
import torch
from module_ConvLSTM import ConvLSTM

class MyConvLSTM(nn.Module):
    def __init__(self):
        super(MyConvLSTM, self).__init__()
        # B, T, C, H, W
        # CNN part for feature extraction
        self.conlstm1 = ConvLSTM(4, 64, (3,3), 1, True, True, False)
        self.conlstm2 = ConvLSTM(64, 128, (5,5), 1, True, True, False)
        self.cnn1 = nn.Conv2d(128,128,3, stride=2, padding=1)
        self.cnn2 = nn.Conv2d(128,64,3, stride=2, padding=1)
        self.cnn3 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.flatten = nn.Flatten() # Flatten the output of CNN
        # Decoder: Fully connected layer
        self.decoder = nn.Linear(1536, 4*12*32) # Output size corresponds to the flattened frame

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        a,_ = self.conlstm1(x)
        b,_ = self.conlstm2(a[0])
        c = self.cnn1(b[0][:,-1,:,:,:])
        d = self.cnn2(c)
        e = self.cnn3(d)
        f = self.flatten(e)
        g = self.decoder(f)
        return g.contiguous().view(batch_size, 4, 12, 32) # Reshape back to image dimensions

if __name__ =='__main__':
    # Create model
    model = MyConvLSTM()

    # Example input
    input_data = torch.randn(8, 4, 4, 12, 32)  # Batch size of 8
    print(input_data.shape)
    output = model(input_data)
    print(output.shape)  # Should print (8, 4, 12, 32)