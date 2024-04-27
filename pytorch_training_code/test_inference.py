import torch
import torch.nn as nn
from models.resnet1d import IMUMinstResNet1D
from models.append_softmax import AppendSoftMax

# Load the trained model
model_dict = torch.load('model.pth')
model = IMUMinstResNet1D(num_classes=10, in_channels=6)
model.load_state_dict(model_dict)

model = AppendSoftMax(model)

input = torch.ones(1, 1800)
output = model(input)
print(output)