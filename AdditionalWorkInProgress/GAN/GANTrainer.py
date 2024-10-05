import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torchvision
from torchvision import datasets, transforms

embed_dim = 128
generator = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    # out: 32 x 64 x 64

    nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()

    #out: 3 x 32 x 32
)

discriminator = nn.Sequential(
    # in: 3 x 32 x 32

    nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 32 x 16 x 16

    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 64 x 8 x 8

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 128 x 4 x 4

    nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0, bias=False),
    # out: 1 x 1 x 1

    nn.Flatten(),
    nn.Sigmoid()
)

def train_generator(generator_optimizer, batch_size, device):
    generator.train()
    # Clear generator gradients
    generator_optimizer.zero_grad()
    
    # Generate fake images
    vector = torch.randn(batch_size, embed_dim, 1,1).to(device) #random noice
    fake_images = generator(vector).to(device) #fake images generated

    #print(fake_images.shape)
    
    # Try to fool the discriminator
    preds = discriminator(fake_images).to(device) #getting the predictions of discriminator for fake images
    targets = torch.ones(batch_size, 1).to(device) #setting 1 as targets so the discriminator can be fooled
    loss = F.binary_cross_entropy(preds, targets) #comparing

    print(f"generator loss: {loss}")
    
    # Update generator weights
    loss.backward()
    generator_optimizer.step()
    
    return loss.item(), vector

def train_discriminator(discriminator_optimizer, input_images, input_targets, batch_size, device):
    discriminator.train()
    # Clear discriminator gradients
    discriminator_optimizer.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(input_images) #real images
    real_preds = real_preds.to(device)

    #real_targets = torch.ones(input_images.size(0), 1).to(device) #setting targets as 1
    real_loss = F.binary_cross_entropy(real_preds, input_targets) #getting the loss
    real_score = torch.mean(real_preds).item()
    
    # Generate fake images
    latent = torch.randn(batch_size, embed_dim, 1, 1).to(device) #generating the random noices for input image
    fake_images = generator(latent).to(device)  #getting the fake images

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1).to(device) #setting 0 as target for fake images
    fake_preds = discriminator(fake_images).to(device)  #getting the predictions for fake images
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)  #Comparing the two scores through loss
    fake_score = torch.mean(fake_preds).item()

    #print(f"real loss {real_loss} | fake loss {fake_loss}")

    # Update discriminator weights
    loss = real_loss + fake_loss

    
    print(f"discriminator loss: {loss}")
    loss.backward()
    discriminator_optimizer.step()

    return loss.item(), real_score, fake_score

def change_target(input):
    if input == 5:
        return 1
    return 0


def train(batch_size, device, lr, loader, generator_optimizer, discriminator_optimizer, discriminator_won):
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    for real_images, targets in loader:
        #print(real_images.shape)
        
        if not discriminator_won:
            real_targets = torch.tensor([[change_target(v)] for v in targets], dtype=torch.float32).to(device)
            #print(real_targets.shape)

            # Train discriminator
            real_images = real_images.to(device)
            loss_d, real_score, fake_score = train_discriminator(discriminator_optimizer, real_images, real_targets, batch_size, device)
        else:
            print("Not training discriminator")
        
        # Train generator
        loss_g, latent = train_generator(generator_optimizer, batch_size, device)
        
    # Record losses & scores
    losses_g.append(loss_g)

    if not discriminator_won:
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

    print(f"lossG {losses_g} | lossD {losses_d}")

    #return losses_g, losses_d, latent, fake_scores

def test(batch_size, device):
    generator.eval()
    discriminator.eval()

    vector = torch.randn(batch_size, embed_dim, 1,1).to(device) #random noice
    fake_images = generator(vector).to(device) #fake images generated
    
    # Try to fool the discriminator
    preds = discriminator(fake_images).to(device) #getting the predictions of discriminator for fake images

    preds_sum = torch.sum(preds)
    preds_avg = preds_sum / len(preds)

    discriminator_won = (preds_avg <= 0.5)
    winner = "discriminator" if preds_avg <= 0.5 else "generator"

    print(f"average prediction: {preds_avg} | current winner: {winner}")
    return discriminator_won

def filter_dataset():
    # Define transformations
    transform = transforms.Compose( #the mean and standard deviation of CIFAR 10
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)

    dog_class_index = dataset.class_to_idx['dog']
    dog_indices = [i for i in range(len(dataset)) if dataset.targets[i] == dog_class_index]

    other_indices = [i for i in range(len(dataset)) if dataset.targets[i] != dog_class_index]
    other_indices = other_indices[:len(dog_indices)]

    filtered_subset = torch.utils.data.Subset(dataset, dog_indices + other_indices)

    #dog_dataset = torch.utils.data.Subset(dataset, dog_indices)
    #other_dataset = torch.utils.data.Subset(dataset, other_indices)

    return filtered_subset#dog_dataset

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=7, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
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


    dog_dataset = filter_dataset()

    generator.to(device=device)
    discriminator.to(device=device)

    '''generator_path = "./GANGenerator.pt"
    discriminator_path = "./GANDiscriminator.pt"

    generator.to(device=torch.device("cuda"))
    loaded_state_dict = torch.load(generator_path)
    generator.load_state_dict(loaded_state_dict)
    
    discriminator.to(device=torch.device("cuda"))
    loaded_state_dict = torch.load(discriminator_path)
    discriminator.load_state_dict(loaded_state_dict)'''


    generator_optimizer = torch.optim.Adam(generator.to(device).parameters(), lr=args.lr/2, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.to(device).parameters(), lr=args.lr, betas=(0.5, 0.999))

    data_loader = torch.utils.data.DataLoader(dog_dataset,**train_kwargs)

    discriminator_won = False
    for epoch in range(1, args.epochs + 1):
        train(args.batch_size, device, args.lr, data_loader, generator_optimizer, discriminator_optimizer, discriminator_won)
        discriminator_won = test(args.batch_size, device)

    if args.save_model:
        torch.save(generator.state_dict(), "GANGenerator.pt")
        torch.save(discriminator.state_dict(), "GANDiscriminator.pt")


if __name__ == '__main__':
    main()
