import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

GRID_SIZE = 32
CANVAS_SIZE = 500

#txt_path = "C:/Users/henry/OneDrive/Documents/GitHub/NeuralNetworks/CurrentWeights.txt"
model_path = "./resnet18Super.pt"
#model_path = "./resnet18.pt"

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            print(data)
            print(data.shape)
            break

            '''#print(target)
            #print(data.shape)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability


            correct += pred.eq(target.view_as(pred)).sum().item()

            print(f"pred {pred} | target {target} | correct {correct}")'''

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    #print(loaded_state_dict)

    transform = transforms.Compose( #the mean and standard deviation of CIFAR 10
        [#transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    #device = torch.device("cuda")

    test_kwargs = {'batch_size': 1}
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': False}
    test_kwargs.update(cuda_kwargs)

    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

    model_path = "./resnet18Super.pt"
    perceptron = torchvision.models.resnet18(num_classes=10)#.to("cuda")
    perceptron = perceptron.to(torch.device("cuda"))
    loaded_state_dict = torch.load(model_path)

    perceptron.load_state_dict(loaded_state_dict)

    correct_count = 0

    test(perceptron, torch.device("cuda"), test_loader)



if __name__ == '__main__':
    main()

print("were done")