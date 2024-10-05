import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from tkinter import *

GRID_SIZE = 32
CANVAS_SIZE = 500


generator_path = "./GANGenerator.pt"
discriminator_path = "./GANDiscriminator.pt"

transform_stats = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
transform = transforms.Compose( #the mean and standard deviation of CIFAR 10
    [transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

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
generator.to(device=torch.device("cuda"))
loaded_state_dict = torch.load(generator_path)
generator.load_state_dict(loaded_state_dict)

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
discriminator.to(device=torch.device("cuda"))
loaded_state_dict = torch.load(discriminator_path)
discriminator.load_state_dict(loaded_state_dict)

device = torch.device("cuda")

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

    dog_dataset = torch.utils.data.Subset(dataset, dog_indices)

    return dog_dataset

dog_dataset = filter_dataset()

class Square:
    def __init__(self, row, col, canvas, r, g, b):
        self.row = row
        self.col = col
        self.canvas = canvas

        mag = CANVAS_SIZE / GRID_SIZE
        self.gui = canvas.create_rectangle(row*mag, col*mag, (row+1)*mag, (col+1)*mag, fill=f"#{r:02x}{g:02x}{b:02x}")
    
    def __str__(self):
        return 1

    def delete(self):
        self.canvas.delete(self.gui)

class App:
    def __init__(self, master):
        frame = Frame(master)
        frame.pack(fill=BOTH, expand=1)

        canvas = Canvas(frame, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas = canvas
        canvas.place(x=0,y=0)

        self.squares = []
        self.raw_data = None

        menu = Frame(frame, width=450, height=500, bg="lightblue")
        menu.place(x=CANVAS_SIZE,y=0)

        rerun_button = Button(menu, text="Rerun", command=self.rerun)
        rerun_button.place(x=20, y=100)

        clear_button = Button(menu, text="Clear", command=self.clear)
        clear_button.place(x=20, y=200)

        load_button = Button(menu, text="Load", command=self.load)
        load_button.place(x=20, y=300)

        self.result = StringVar()
        status = Label(menu, textvariable=self.result, wraplength=250, font=("Arial", 15))
        status.place(x=150, y=350)
    
    def draw_square(self, x, y, r, g, b):
        if x <= CANVAS_SIZE and x >= 0 and y <= CANVAS_SIZE and y >= 0:
            row = int(x / CANVAS_SIZE * GRID_SIZE)
            col = int(y / CANVAS_SIZE * GRID_SIZE)

            square = Square(row, col, self.canvas, r, g, b)
            self.squares.append(square)
            
            #self.rerun()

    def rerun(self):
        generator.eval()
        discriminator.eval()

        with torch.no_grad():
            vector = torch.randn(1, embed_dim, 1,1).to(device)
            fake_image = generator(vector).to(device) #fake images generated

            #denormalize
            image = fake_image * transform_stats[1][0] + transform_stats[0][0]

            '''for i_row in range(len(data)):
                for i_v in range(len(data[i_row])):
                    rgb = data[i_v][i_row]

                    self.draw_square(i_row*cell_size, i_v*cell_size, rgb[0], rgb[1], rgb[2])'''
            image = image[0]
            #image.cpu()
            #print(image)
            #print(image.shape)

            raw_image = image.cpu().numpy().transpose(1, 2, 0)

            # Clip values to [0, 1]
            raw_image = np.clip(raw_image, 0, 1)

            # Convert to uint8
            raw_image = (raw_image * 255).astype(np.uint8)

            #print(raw_image)
            print(raw_image.shape)

            #Display image
            cell_size = CANVAS_SIZE / GRID_SIZE
            for i_row in range(len(raw_image)):
                for i_v in range(len(raw_image[i_row])):
                    rgb = raw_image[i_v][i_row]

                    self.draw_square(i_row*cell_size, i_v*cell_size, rgb[0], rgb[1], rgb[2])
            
            preds = discriminator(fake_image).to(device)
            answer = "not a dog" if preds.data <= 0.5 else "a dog"
            
            self.result.set(f"Discriminator Prediction: {preds.data}, it thinks that this image is {answer}")
    
    def load(self):
        pass

    def clear(self):
        for square in self.squares:
            square.delete()
        
        squares = []


root = Tk()
root.minsize(CANVAS_SIZE+450, CANVAS_SIZE)
root.resizable(False, False)
root.wm_title("CIFAR-10 GAN Generator")

app = App(root)

root.mainloop()

print("were done")