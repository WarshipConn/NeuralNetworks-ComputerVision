import numpy as np
import torch
import torch.nn as nn
from tkinter import *

GRID_SIZE = 20
CANVAS_SIZE = 500

txt_path = "C:/Users/henry/OneDrive/Documents/GitHub/NeuralNetworks/CurrentWeights.txt"

grid = np.array([[0]*GRID_SIZE]*GRID_SIZE)
print(grid)

class NeuronLayer:
    def __init__(self, w, b):
        self.w = torch.tensor(w, requires_grad=True, dtype=float)
        self.b = torch.tensor(b, requires_grad=True, dtype=float)

def make_neuron_layer(input_size, output_size):
    mag = 5#np.sqrt(6) / np.sqrt(input_size + output_size)

    w = [[(np.random.rand() * mag * 2 - mag) for i in range(input_size)] for j in range(output_size)]
    #w = [[1]*input_size]*output_size
    b = [mag - np.random.rand() * 2 * mag]*output_size

    return NeuronLayer(w, b)

class NeuralNetwork:
    def __init__(self, input_size, output_size, layers):
        #output_size = 1 #always 1 for now

        self.output_size = output_size
        self.input_size = input_size
        self.layers = layers

        self.error = 0

        '''self.layer_list = [
            NeuronLayer(torch.tensor([[3,4],[2,3]], requires_grad=True, dtype=float), torch.tensor([-2,-4], requires_grad=True, dtype=float)), 
            NeuronLayer(torch.tensor([[5,-5]], requires_grad=True, dtype=float), torch.tensor([-2], requires_grad=True, dtype=float))
        ]'''
        
        self.layer_list = [make_neuron_layer(input_size, input_size)]*(layers-1) + [make_neuron_layer(input_size, output_size)]
    
    def load(self, fileDir):
        self.layer_list = []

        file = open(fileDir, "r")
        read_weights = file.read().splitlines()
        mode = None

        weight_arr = []
        weight_row = []
        bias_arr = []

        for i in read_weights:
            if i == "##":
                self.layer_list += [NeuronLayer(weight_arr, bias_arr)]

                weight_arr = []
                weight_row = []
                bias_arr = []
            elif i == "#w":
                mode = "w"
            elif i == "#b":
                mode = "b"
            else:
                if mode == "w":
                    if i == "#":
                        weight_arr += [weight_row]
                        weight_row = []
                    else:
                        weight_row += [float(i)]
                elif mode == "b":
                    bias_arr += [float(i)]
        
        file.close()
        print("load complete")
    
    def forward(self, x):
        y = x

        for layer in self.layer_list:
            s = layer.w.mv(y) + layer.b
            y = torch.nn.Sigmoid()(s)
        return y

    def output(self, x):
        y = self.forward(x)
        return y.detach().numpy()

perceptron = NeuralNetwork(GRID_SIZE*GRID_SIZE, 10, 10)
perceptron.load(txt_path)

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
        self.canvas.coords(self.gui, 0, BAR_MAX_Y*self.index + BAR_MARGIN*self.index, BAR_MAX_X * size / max_size, BAR_MAX_Y*(self.index + 1) + BAR_MARGIN*self.index)

        if size == max_size:
            self.canvas.itemconfig(self.gui, fill="red")
        else:
            self.canvas.itemconfig(self.gui, fill="black")


class App:
    def __init__(self, master):
        frame = Frame(master)
        frame.pack(fill=BOTH, expand=1)

        canvas = Canvas(frame, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas = canvas
        canvas.grid(row=0, column=0, sticky=W+E+N+S)

        self.squares = []
        self.drawing = False
        canvas.bind("<Button-1>", self.mouse_down)
        canvas.bind("<B1-Motion>", self.mouse_move)
        canvas.bind("<ButtonRelease-1>", self.mouse_up)

        menu = Frame(frame, bg="lightgreen")
        menu.grid(row=0, column=1, sticky=W+E+N+S)

        rerun_button = Button(menu, text="Rerun", command=self.rerun)
        rerun_button.grid(row=0, column=0)

        clear_button = Button(menu, text="Clear", command=self.clear)
        clear_button.grid(row=1, column=0)

        self.result = StringVar()
        status = Label(menu, textvariable=self.result)
        status.grid(row=1, column=1)


        result_canvas = Canvas(menu, width=300, height=300)
        self.result_canvas = result_canvas
        result_canvas.grid(row=0, column=1)

        self.bars = []
        for i in range(10):
            self.bars += [Bar(i, 0, 1, result_canvas)]
        

    
    def draw_square(self, x, y):
        if x < CANVAS_SIZE and x > 0 and y < CANVAS_SIZE and y > 0:
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
        #print(grid.reshape(-1))

        result = perceptron.output(torch.tensor(grid.reshape(-1), dtype=torch.double))
        result_max = np.max(result)

        print(result)
        answer = np.where(result == result_max)

        for i in range(len(self.bars)):
            self.bars[i].update(result[i], result_max)

        self.result.set(f"Outputs updated, best guesses: {answer}")
    
    def clear(self):
        for square in self.squares:
            square.delete()
        
        squares = []

        self.result.set("Canvas cleared")
        print("clear")
    


root = Tk()
root.minsize(CANVAS_SIZE+300, CANVAS_SIZE)
root.resizable(False, False)
root.wm_title("Perceptron Reader")

app = App(root)

root.mainloop()


