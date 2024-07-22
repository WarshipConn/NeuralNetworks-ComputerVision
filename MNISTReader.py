import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tkinter import *

GRID_SIZE = 28
CANVAS_SIZE = 500

txt_path = "C:/Users/henry/OneDrive/Documents/GitHub/NeuralNetworks/CurrentWeights.txt"
model_path = "C:/Users/henry/OneDrive/Documents/GitHub/NeuralNetworks/mnist_cnn.pt"

grid = np.array([[0]*GRID_SIZE]*GRID_SIZE)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=-1)
        return output

    def output(self, x):
        return torch.exp(self.forward(x))

perceptron = Net()
loaded_state_dict = torch.load(model_path)

perceptron.load_state_dict(loaded_state_dict)

#print(loaded_state_dict)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) #the mean and standard deviation of the pixel intensities in the MNIST dataset
])

dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

class Square:
    def __init__(self, row, col, canvas):
        self.row = row
        self.col = col
        self.canvas = canvas

        mag = CANVAS_SIZE / GRID_SIZE
        self.gui = canvas.create_rectangle(row*mag, col*mag, (row+1)*mag, (col+1)*mag, fill="black")

        grid[col][row] = 1
    
    def __str__(self):
        return 1

    def delete(self):
        self.canvas.delete(self.gui)
        grid[self.col][self.row] = 0

BAR_MAX_X = 250
BAR_MAX_Y = 20
BAR_MARGIN = 10

class Bar:
    def __init__(self, index, size, max_size, canvas):
        self.index = index
        self.canvas = canvas

        self.gui = canvas.create_rectangle(0,0,0,0, fill="black")
    
    def update(self, size, max_size):
        self.canvas.coords(self.gui, 0, BAR_MAX_Y*self.index + BAR_MARGIN*self.index, 10 + BAR_MAX_X * size / max_size, BAR_MAX_Y*(self.index + 1) + BAR_MARGIN*self.index)

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
        self.drawing = False
        canvas.bind("<Button-1>", self.mouse_down)
        canvas.bind("<B1-Motion>", self.mouse_move)
        canvas.bind("<ButtonRelease-1>", self.mouse_up)

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
        status = Label(menu, textvariable=self.result, wraplength=200)
        status.place(x=150, y=350)
        #status.grid(row=1, column=1)

        result_canvas = Canvas(menu, width=300, height=300)
        self.result_canvas = result_canvas
        result_canvas.place(x=150, y=20)
        #result_canvas.grid(row=0, column=1)

        for i in range(10):
            label = Label(menu, text=str(i))
            label.place(x=125, y=20+i*30)

        self.bars = []
        for i in range(10):
            self.bars += [Bar(i, 0, 1, result_canvas)]
        

    
    def draw_square(self, x, y):
        if x <= CANVAS_SIZE and x >= 0 and y <= CANVAS_SIZE and y >= 0:
            row = int(x / CANVAS_SIZE * GRID_SIZE)
            col = int(y / CANVAS_SIZE * GRID_SIZE)

            if grid[col][row] == 0:
                square = Square(row, col, self.canvas)
                self.squares.append(square)
            
            self.rerun()
            


    def mouse_down(self, event):
        self.drawing = True
        self.draw_square(event.x, event.y)
        pass

    def mouse_move(self, event):
        if self.drawing:
            self.draw_square(event.x, event.y)

    def mouse_up(self, event):
        self.drawing = False
        pass

    def rerun(self):
        result = perceptron.output(torch.tensor([grid], dtype=torch.float))
        np_result = result.data.numpy()

        result_max = np.max(np_result)

        answer = np.where(np_result == result_max)

        for i in range(len(self.bars)):
            self.bars[i].update(np_result[i], result_max)

        self.result.set(f"Outputs updated, best guesses: {answer}")
    
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

        rand_i = int(np.random.rand()*60000)

        data = dataset.data[rand_i].data.numpy()
        target = dataset.targets.data[rand_i]

        for i_row in range(len(data)):
            for i_v in range(len(data[i_row])):
                if data[i_v][i_row] > 0:
                    self.draw_square(i_row*cell_size, i_v*cell_size)
        
        self.result.set(f"Currently Displaying: {target}")
    


root = Tk()
root.minsize(CANVAS_SIZE+450, CANVAS_SIZE)
root.resizable(False, False)
root.wm_title("MNIST Perceptron Reader")

app = App(root)

root.mainloop()


print("were done")