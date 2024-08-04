import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class ResNetBlock(nn.Module):
    def __init__(self, channel_in, channel_out, stride, downsample):
        super(ResNetBlock, self).__init__() 

        self.conv1 = nn.Conv2d(channel_in, channel_out, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(channel_out)
        self.conv2 = nn.Conv2d(channel_out, channel_out, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channel_out)

        self.dropout = nn.Dropout(0.3) #Add some dropouts to help prevent overfitting

        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        i = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        #print(f"y {y.shape} | i {i.shape}")

        y = self.conv2(y)
        y = self.bn2(y)

        #print(f"y {y.shape} | i {i.shape}")

        if self.downsample is not None:
            i = self.downsample(x)

        #print(f"y {y.shape} | i {i.shape}")

        y += i
        y = self.relu(y)
        y = self.dropout(y)
        return y


class ResNet(nn.Module):
    def __init__(self, img_channel, classes_amt):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(img_channel, 64, 7, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, classes_amt)

        self.dropout = nn.Dropout(0.25) #Add some dropouts to help prevent overfitting

        self.layers = [
            self.make_layer(64, 64, 1),
            self.make_layer(64, 128, 2),
            self.make_layer(128, 256, 2),
            self.make_layer(256, 512, 2)
        ]
    
    def make_layer(self, in_channel, out_channel, stride):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride),
                nn.BatchNorm2d(out_channel)
            )

        block = ResNetBlock(in_channel, out_channel, stride, downsample)
        block = block.to(torch.device("cuda"))

        return block


    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        for layer in self.layers:
            #print("running block")
            y = layer.forward(y)
        
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        y = F.softmax(y, 1)

        return y

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target) #Cross entropy loss function for resnet18
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

final_accuracies = []

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=7, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda") #Using CUDA
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose( #the mean and standard deviation of CIFAR 10
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    ) 
    dataset1 = datasets.CIFAR10('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR10('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = ResNet(3, 10).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    #Tests done with ~5 epochs each, since anything else beyond that just plateaus

    #optim.Adadelta(model.parameters(), lr=args.lr) #Around 30% to 40%, increase by around 2 or 3% per epoch
    #optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) #Always 10% for some reason, no improvements
    #optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) #Around 15% to 25%, increase by around 1 or 2% per epoch


    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    #StepLR(optimizer, step_size=1, gamma=args.gamma)
    #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(test_loader), eta_min=0) #Always performs worse

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "resnet18.pt")


if __name__ == '__main__':
    main()
