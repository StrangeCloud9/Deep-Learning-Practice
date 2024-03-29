import config
import paths

import torch
import torch.nn as nn


class FFNN(nn.Sequential):

    def __init__(self):
        l = []
        sizes = [config.input_size] + config.hidden + [config.nclasses]
        for i in range(len(sizes)-2):
            l.append(nn.Linear(sizes[i], sizes[i+1]))
            if config.dropout > 0:
                l.append(nn.Dropout(config.dropout))
            l.append(nn.ReLU())
        l.append(nn.Linear(sizes[-2], sizes[-1]))
        super(FFNN, self).__init__(*l)

    @staticmethod
    def load():
        model = FFNN()
        path = paths.model
        model.load_state_dict(torch.load(path))

        return model

    def save(self):
        path = paths.model
        torch.save(self.state_dict(), path)

class MLP(nn.Module):
    
    def __init__(self, input_dim = 0, hidden_dim = 0, output_dim = 0, nlayer = 0, CUDA = False):

        super(MLP, self).__init__() 
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 
        self.nlayer = nlayer 
        self.CUDA = CUDA

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.layer4 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.bn4 = nn.BatchNorm1d(int(hidden_dim))
        
        self.layer5 = nn.Linear(int(hidden_dim/2), int(hidden_dim/2))
        self.bn5 = nn.BatchNorm1d(int(hidden_dim/2))
        
        self.bn_out = nn.BatchNorm1d(int(hidden_dim / 2))
        self.output_layer = nn.Linear(int(hidden_dim/2), output_dim)        
        
        return 

    def forward(self, x):

        layer1 = self.layer1(self.bn1(x))
        layer1 = layer1.cuda() if self.CUDA else layer1 

        layer2 = self.layer2(self.bn2(layer1))
        layer2 = layer2.cuda() if self.CUDA else layer2

        layer3 = self.layer3(self.bn3(layer2))
        layer3 = layer3.cuda() if self.CUDA else layer3

        layer4 = self.layer4(self.bn4(layer3))
        layer4 = layer4.cuda() if self.CUDA else layer4

        layer5 = self.layer5(self.bn5(layer4))
        layer5 = layer5.cuda() if self.CUDA else layer5

        layer6 = self.layer6(self.bn6(layer5))
        layer6 = layer6.cuda() if self.CUDA else layer6

        output = self.output_layer(self.bn_out(layer6))
        
        return output

    @staticmethod
    def load():
        model = MLP()
        path = paths.model
        model.load_state_dict(torch.load(path))

        return model

    def save(self):
        path = paths.model
        torch.save(self.state_dict(), path)
