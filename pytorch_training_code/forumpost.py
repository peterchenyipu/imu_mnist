import torch
import torch.nn as nn
import onnx
import onnxruntime

class SimplifiedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_rate=0.0):
        super(SimplifiedResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x += shortcut
        x = self.relu(x)
        
        return x
    
class SmallResNet1DWithReshape(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super(SmallResNet1DWithReshape, self).__init__()
        self.conv1 = nn.Conv1d(6, 32, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        
        self.layer1 = SimplifiedResidualBlock(32, 32, dropout_rate=dropout_rate)
        self.layer2 = SimplifiedResidualBlock(32, 64, stride=2, dropout_rate=dropout_rate)
        self.layer3 = SimplifiedResidualBlock(64, 128, stride=2, dropout_rate=dropout_rate)
        self.layer4 = SimplifiedResidualBlock(128, 256, stride=2, dropout_rate=dropout_rate)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # B = x.shape[0]
        x = x.reshape(-1, 300, 6)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
model = SmallResNet1DWithReshape(num_classes=10, dropout_rate=0.0)
# model.eval()

# Export the model to ONNX, fix the batch size to 1
dummy_input = torch.ones(1, 1800)
print(f'model inference result: {model(dummy_input)}')

torch.onnx.export(model, dummy_input,
                  'model.onnx',
                  opset_version=12,
                  input_names=['input'],
                  output_names=['output'])

# load the model and run inference
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession('model.onnx')
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)

print(f'onnx inference result: {ort_outs[0]}')
