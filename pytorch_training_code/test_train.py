from dataset.ei_dataset import EdgeImpulseDataset
from models.resnet1d import IMUMinstResNet1D
import torch
from torchinfo import summary
from tqdm import tqdm

if __name__ == '__main__':
        
    net = IMUMinstResNet1D(num_classes=10, in_channels=6, dropout_rate=.1)
    print(summary(net, (10, 1800), device='cpu'))
    
    net.to('cuda')
    train_dataset = EdgeImpulseDataset('data/EI_dataset/training', split='training', split_ratio=0.8)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    vali_dataset = EdgeImpulseDataset('data/EI_dataset/testing', split='validation', split_ratio=0.8)
    vali_dataloader = torch.utils.data.DataLoader(vali_dataset, batch_size=32, shuffle=False)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    
    best_accuracy = 0.0  # Initialize the best accuracy to 0

    for epoch in tqdm(range(300)):
        net.train()
        for i, (data, label) in enumerate(train_dataloader):
            data, label = data.to('cuda'), label.to('cuda')
            optimizer.zero_grad()
            output = net(data)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            
            # if i % 10 == 0:
        print(f'Epoch {epoch}, loss: {loss.item()}')
            
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, label in vali_dataloader:
                data, label = data.to('cuda'), label.to('cuda')
                output = net(data)
                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                
                
        accuracy = correct / total
        print(f'Epoch {epoch}, Validation Accuracy: {accuracy}')
        
        # Save the model if it has the best accuracy so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(net.state_dict(), 'best_model.pth')
            
    # Load the best model
    net.load_state_dict(torch.load('best_model.pth'))
    # check the accuracy on the test set
    test_dataset = EdgeImpulseDataset('data/EI_dataset/testing')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in test_dataloader:
            data, label = data.to('cuda'), label.to('cuda')
            output = net(data)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy}')
                
    print('Finished training')
    torch.save(net.state_dict(), 'model.pth')
