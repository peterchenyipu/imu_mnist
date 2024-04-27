import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=1):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
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

def resnet18_1d(**kwargs):
    """Constructs a ResNet-18 model for 1D data."""
    model = ResNet1D(BasicBlock1D, [2, 2, 2, 2], **kwargs)
    return model

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
        # shortcut = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        # x += shortcut
        x = self.relu(x)
        
        return x
    
class SmallResNet1D(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super(SmallResNet1D, self).__init__()
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
    
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv1d(6, 32, 3)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)
        
    
    def forward(self, x):
        x = x.reshape(-1, 300, 6).transpose(1, 2)
        x = self.conv1(x) # B x 32 x 298
        x = self.maxpool(x) # B x 32 x 149
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # B x 32*298
        
        x = self.fc(x)
        return x

class SmallResNet1D1Layer(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super(SmallResNet1D1Layer, self).__init__()
        self.conv1 = nn.Conv1d(6, 32, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        
        self.layer1 = SimplifiedResidualBlock(32, 32, dropout_rate=dropout_rate)
        self.layer2 = SimplifiedResidualBlock(32, 64, stride=2, dropout_rate=dropout_rate)
        self.layer3 = SimplifiedResidualBlock(64, 128, stride=2, dropout_rate=dropout_rate)
        self.layer4 = SimplifiedResidualBlock(128, 256, stride=2, dropout_rate=dropout_rate)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # B = x.shape[0]
        x = x.reshape(-1, 300, 6).transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class ThreeLayerConv1D(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0, in_channels=6):
        super(ThreeLayerConv1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        
        self.layer1 = SimplifiedResidualBlock(32, 32, dropout_rate=dropout_rate)
        self.layer2 = SimplifiedResidualBlock(32, 64, stride=2, dropout_rate=dropout_rate)
        # self.layer3 = SimplifiedResidualBlock(64, 128, stride=2, dropout_rate=dropout_rate)
        # self.layer4 = SimplifiedResidualBlock(128, 256, stride=2, dropout_rate=dropout_rate)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class IMUMinstResNet1D(nn.Module):
    def __init__(self, num_classes=10, in_channels=6, dropout_rate=0.0):
        super(IMUMinstResNet1D, self).__init__()
        
        # self.resnet = resnet18_1d(num_classes=num_classes, in_channels=in_channels)
        self.resnet = ThreeLayerConv1D(num_classes=num_classes, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): input tensor of shape B x 1800
        """
        
        # Reshape the input tensor to B x 6 x 300, note 1d tensor is listed by rows, so we need to transpose
        # x = x.view(-1, 300, 6).transpose(1, 2)
        x = x.reshape(-1, 300, 6).transpose(1, 2)
        # print(x)
        return self.resnet(x)

class AccelMINSTConv(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super(AccelMINSTConv, self).__init__()
        
        self.conv_net = ThreeLayerConv1D(num_classes=num_classes, dropout_rate=dropout_rate, in_channels=3)
    
    def forward(self, x):
        x = x.reshape(-1, 300, 3).transpose(1, 2)
        return self.conv_net(x)
        



if __name__ == '__main__':
    from torchinfo import summary
    
    net = IMUMinstResNet1D(num_classes=10, in_channels=6, dropout_rate=0.5)
    summary(net, (1, 1800), device='cpu')
    print()
    net2 = AccelMINSTConv(num_classes=10, dropout_rate=0.5)
    summary(net2, (1, 900), device='cpu')
    