import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from vit_pytorch import ViT

model_path = "./ViT.pt"

perceptron = ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 128,
    depth = 2,
    heads = 16,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)
perceptron.to(device=torch.device("cuda"))
loaded_state_dict = torch.load(model_path)
perceptron.load_state_dict(loaded_state_dict)


transform = transforms.Compose( #the mean and standard deviation of CIFAR 10
    [#transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)
dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
class_names = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

rand_i = int(np.random.rand()*10000)
data = dataset.data[rand_i]
target = dataset.targets[rand_i]

print(f"Currently Class: {class_names[target]}")

perceptron.eval()
with torch.no_grad():
    transformed_image = transforms.ToTensor()(data)
    x = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(transformed_image)

    x = torch.tensor(np.array([x]), dtype=torch.float)
    x = x.to(torch.device("cuda"))


    embedded_vector = perceptron.embed(x)

    print(embedded_vector)
    print(embedded_vector.shape)
