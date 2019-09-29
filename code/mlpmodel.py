import config
import paths
import torch.nn.functional as F
import torch
import torch.nn as nn
class MLP(nn.Module):
    
    def __init__(self, input_dim = 0, hidden_dim = 0, output_dim = 0, CUDA = False):

        super(MLP, self).__init__() 
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 
        self.CUDA = CUDA

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.layer4 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.bn4 = nn.BatchNorm1d(int(hidden_dim/2))
        
        self.layer5 = nn.Linear(int(hidden_dim/2), int(hidden_dim/2))
        self.bn5 = nn.BatchNorm1d(int(hidden_dim/2))
        
        
        self.output_layer = nn.Linear(int(hidden_dim/2), output_dim)        
        self.bn_out = nn.BatchNorm1d(int(output_dim))

        return 

    def forward(self, x):
        
        layer1 = self.layer1(x)
        layer1 = F.relu(layer1)
        layer1 = self.bn1(layer1)
        layer1 = layer1.cuda() if self.CUDA else layer1 

        layer2 = self.layer2(layer1)
        layer2 = F.relu(layer2)
        layer2 = self.bn2(layer2)
        layer2 = layer2.cuda() if self.CUDA else layer2 

        layer3 = self.layer3(layer2)
        layer3 = F.relu(layer3)
        layer3 = self.bn3(layer3)
        layer3 = layer3.cuda() if self.CUDA else layer3 

        layer4 = self.layer4(layer3)
        layer4 = F.relu(layer4)
        layer4 = self.bn4(layer4)
        layer4 = layer4.cuda() if self.CUDA else layer4 

        layer5 = self.layer5(layer4)
        layer5 = F.relu(layer5)
        layer5 = self.bn5(layer5)
        layer5 = layer5.cuda() if self.CUDA else layer5

        output = self.output_layer(layer5)
        
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