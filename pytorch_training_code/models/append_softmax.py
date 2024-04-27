import torch.nn as nn

class AppendSoftMax(nn.Module):
    def __init__(self, model):
        super(AppendSoftMax, self).__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x