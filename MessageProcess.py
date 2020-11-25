import torch
import torch.nn as nn

class MLP_decode(nn.Module):
    def __init__(self):
        super(MLP_decode, self).__init__()
        self.fc1 = nn.Linear(64*64, 128*128)
        self.fc2 = nn.Linear(128*128, 128 * 128)
        # self.drop1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(128*128, 64*64)
        # self.drop2 = nn.Dropout(0.6)
        self.fc3 = nn.Linear(64*64, 16*16)
        self.fc4 = nn.Linear(16*16, 64)

    def forward(self, d):
        din = d.reshape(d.shape[0],64*64)
        dout1 = nn.functional.elu(self.fc1(din))
        dout2 = nn.functional.elu(self.fc2(dout1))
        dout3 = nn.functional.elu(self.fc3(dout2))
        dout4 = self.fc4(dout3)
        return dout4

class MLP_encode(nn.Module):
    def __init__(self):
        super(MLP_encode, self).__init__()
        self.fc1 = nn.Linear(64*1, 16*16)
        self.drop1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(16*16, 32*32)
        self.drop2 = nn.Dropout(0.6)
        self.fc3 = nn.Linear(32*32, 64*64)

    def forward(self, din):
        dout1 = nn.functional.elu(self.drop1(self.fc1(din)))
        dout2 = nn.functional.elu(self.drop2(self.fc2(dout1)))
        dout3 = self.fc3(dout2)
        dout = dout3.reshape(din.shape[0],1,64,64)
        return dout

#
# model_drop = MLP().cuda()
# print(model_drop)
# optimizer = optim.SGD(model_drop.parameters(), lr=0.01, momentum=0.9)
# lossfunc = nn.CrossEntropyLoss().cuda()