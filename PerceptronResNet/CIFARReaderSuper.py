import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from tkinter import *

GRID_SIZE = 32
CANVAS_SIZE = 500

#txt_path = "C:/Users/henry/OneDrive/Documents/GitHub/NeuralNetworks/CurrentWeights.txt"
model_path = "./resnet18Super.pt"
#model_path = "./resnet18.pt"


perceptron = torchvision.models.resnet18(num_classes=10)#.to("cuda")
perceptron = perceptron.to(torch.device("cuda"))
loaded_state_dict = torch.load(model_path)

perceptron.load_state_dict(loaded_state_dict)

#print(loaded_state_dict)

transform = transforms.Compose( #the mean and standard deviation of CIFAR 10
    [#transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

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
            perceptron.eval()
            with torch.no_grad():
                x = self.raw_data

                #print(x)

                transformed_image = transforms.ToTensor()(x)
                x = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(transformed_image)

                x = torch.tensor(np.array([x]), dtype=torch.float)
                x = x.to(torch.device("cuda"))

                print(x)

                #print(x.shape)
                result = perceptron(x).cpu()
                answer = result.argmax(dim=1, keepdim=True)
                
                
                softmax_result = F.softmax(result, dim=1)
                np_result = softmax_result.data.numpy()[0]

                result_max = np_result[answer[0][0]]

                print(softmax_result)

                print(result_max)

                for i in range(len(self.bars)):
                    self.bars[i].update(np_result[i], result_max)
                    pass

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