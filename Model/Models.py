from torch.nn.modules.batchnorm import BatchNorm1d
import torch.nn as nn
import torch

class Model(torch.nn.Module):
    def __init__(self, context=0):
        super(Model, self).__init__()
        # TODO: Please try different architectures
        in_size = 13 * (context * 2 + 1)
        layers = [
            nn.Linear(in_size, 2048, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048, bias=True),
            ######################
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048, bias=True)

        ]
        # layer2=[
        #     nn.Linear(2048,2048,bias=True),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.BatchNorm1d(2048),
        #     nn.Linear(2048,2048,bias=True),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.BatchNorm1d(2048),
        #     nn.Linear(2048,2048,bias=True),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.BatchNorm1d(2048),
        #     nn.Linear(2048,2048,bias=True),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),

        # ]
        # resnet=[
        #     nn.BatchNorm1d(2048),
        #     nn.Linear(2048,2048,bias=True),    
        # ]
        self.laysers = nn.Sequential(*layers)
        # self.layser2 = nn.Sequential(*layer2)
        # self.resnet=nn.Sequential(*resnet)

    def forward(self, A0):
        res = self.laysers(A0)
        # y = self.layser2(x)
        # res=self.resnet(x+y)
        # res=self.resnet(x)

        return res