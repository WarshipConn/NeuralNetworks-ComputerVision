import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tkinter import *

GRID_SIZE = 32
CANVAS_SIZE = 500

#txt_path = "C:/Users/henry/OneDrive/Documents/GitHub/NeuralNetworks/CurrentWeights.txt"
#model_path = "./resnet18Super.pt"
model_path = "./resnet18.pt"

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
        #y = self.bn1(y)
        y = self.relu(y)

        #print(f"y {y.shape} | i {i.shape}")

        y = self.conv2(y)
        #y = self.bn2(y)

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
                #nn.BatchNorm2d(out_channel)
            )

        block = ResNetBlock(in_channel, out_channel, stride, downsample)
        block = block.to(torch.device("cuda"))

        return block


    def forward(self, x):

        x = x.to(torch.device("cuda"))

        y = self.conv1(x)
        #y = self.bn1(y)
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

perceptron = ResNet(3, 10).to("cuda")
loaded_state_dict = torch.load(model_path)

perceptron.load_state_dict(loaded_state_dict)

#print(loaded_state_dict)

transform = transforms.Compose( #the mean and standard deviation of CIFAR 10
    [transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)

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

BAR_MAX_X = 250
BAR_MAX_Y = 20
BAR_MARGIN = 10

class Bar:
    def __init__(self, index, size, max_size, canvas):
        self.index = index
        self.canvas = canvas

        self.gui = canvas.create_rectangle(0,0,0,0, fill="black")
    
    def update(self, size, max_size):       

        x1 = 0
        y1 = BAR_MAX_Y*self.index + BAR_MARGIN*self.index
        x2 = 10 + BAR_MAX_X * size / max_size
        y2 = BAR_MAX_Y*(self.index + 1) + BAR_MARGIN*self.index

        self.canvas.coords(self.gui, x1, y1, x2, y2)

        if size == max_size:
            self.canvas.itemconfig(self.gui, fill="red")
        else:
            self.canvas.itemconfig(self.gui, fill="black")
    
    def reset(self):
        self.canvas.coords(self.gui, 0, 0, 0, 0)


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
        #rerun_button.grid(row=0, column=0)

        clear_button = Button(menu, text="Clear", command=self.clear)
        clear_button.place(x=20, y=200)
        #clear_button.grid(row=1, column=0)

        load_button = Button(menu, text="Load", command=self.load)
        load_button.place(x=20, y=300)

        self.result = StringVar()
        status = Label(menu, textvariable=self.result, wraplength=250, font=("Arial", 15))
        status.place(x=150, y=350)

        self.guess = StringVar()
        guess_label = Label(menu, textvariable=self.guess, wraplength=250, font=("Arial", 15))
        guess_label.place(x=150, y=400)
        #status.grid(row=1, column=1)

        result_canvas = Canvas(menu, width=300, height=300)
        self.result_canvas = result_canvas
        result_canvas.place(x=150, y=20)
        #result_canvas.grid(row=0, column=1)

        self.class_names = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
        for i in range(10):
            label = Label(menu, text=str(self.class_names[i]))
            label.place(x=75, y=20+i*30)

        self.bars = []
        for i in range(10):
            self.bars += [Bar(i, 0, 1, result_canvas)]
        

    
    def draw_square(self, x, y, r, g, b):
        if x <= CANVAS_SIZE and x >= 0 and y <= CANVAS_SIZE and y >= 0:
            row = int(x / CANVAS_SIZE * GRID_SIZE)
            col = int(y / CANVAS_SIZE * GRID_SIZE)

            square = Square(row, col, self.canvas, r, g, b)
            self.squares.append(square)
            
            #self.rerun()

    def rerun(self):
        if self.raw_data is not None:
            x = torch.tensor(np.array([self.raw_data]), dtype=torch.float)
            x = x.permute(0, 3, 1, 2)
            perceptron.eval()
            result = perceptron(x).cpu()
            np_result = result.data.numpy()[0]

            result_max = np.max(np_result)

            answer = np.where(np_result == result_max)

            for i in range(len(self.bars)):
                self.bars[i].update(np_result[i], result_max)

            self.guess.set(f"Outputs updated, best guesses: {[self.class_names[target[0]] for target in answer]}")
        else:
            self.guess.set(f"Missing input")
    
    def clear(self):
        for square in self.squares:
            square.delete()
        
        squares = []

        for bar in self.bars:
            bar.reset()

        self.result.set("Canvas cleared")
        print("clear")
    
    def load(self):
        self.clear()
        cell_size = CANVAS_SIZE / GRID_SIZE

        rand_i = int(np.random.rand()*10000)

        data = dataset.data[rand_i]
        target = dataset.targets[rand_i]

        self.raw_data = data

        #print(len(data[0][0]))

        for i_row in range(len(data)):
            for i_v in range(len(data[i_row])):
                rgb = data[i_v][i_row]

                self.draw_square(i_row*cell_size, i_v*cell_size, rgb[0], rgb[1], rgb[2])
        
        self.result.set(f"Currently Displaying: {self.class_names[target]}")

        self.rerun()
    


root = Tk()
root.minsize(CANVAS_SIZE+450, CANVAS_SIZE)
root.resizable(False, False)
root.wm_title("CIFAR-10 Perceptron Reader")

app = App(root)

root.mainloop()

print("were done")