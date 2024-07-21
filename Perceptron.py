import numpy as np
import torch
import torch.nn as nn

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
    
    def save(self, fileDir):
        file = open(fileDir, "w")

        for layer in self.layer_list:
            np_w = layer.w.detach().numpy()
            np_b = layer.b.detach().numpy()

            file.write("#w\n")
            if np_w.ndim == 1:
                for w in w_r:
                    file.write(str(w)+"\n")
                file.write("#\n")
            else:
                for w_r in np_w:
                    for w in w_r:
                        file.write(str(w)+"\n")
                    file.write("#\n")

            file.write("#b\n")
            for b in np_b:
                file.write(str(b)+"\n")
            
            file.write("##\n")

        file.close()
        print("save complete")

    def forward(self, x):
        y = x

        for layer in self.layer_list:
            s = layer.w.mv(y) + layer.b
            y = torch.nn.Sigmoid()(s)
        return y
    
    def train_once(self, x, t, mu):
        y = self.forward(x)

        loss_fn = nn.MSELoss()

        loss = loss_fn(y, t)
        loss.backward()

        #print(loss.data.numpy())

        #error = 1/2 * (t - y) ** 2

        #print(error.data.numpy()[0])
        self.error = loss.data.numpy()

        #error.backward()

        for layer in self.layer_list:
            layer.w.data = layer.w - mu * layer.w.grad.data
            layer.b.data = layer.b - mu * layer.b.grad.data

            layer.w.grad.data.zero_()
            layer.b.grad.data.zero_()
        
    def output(self, x):
        y = self.forward(x)
        return y.detach().numpy()


txt_path = "C:/Users/henry/OneDrive/Documents/GitHub/NeuralNetworks/CurrentWeights.txt"
data_path = "C:/Users/henry/OneDrive/Documents/GitHub/NeuralNetworks/PerceptronTrainingData.txt"

GRID_SIZE = 20

perceptron = NeuralNetwork(GRID_SIZE*GRID_SIZE, 10, 5)
perceptron.load(txt_path)

training_data = []

def load_training_data():
    file = open(data_path, "r")
    read_info = file.read().splitlines()
    mode = None

    data = []

    training_x = []
    training_y = None

    for l in read_info:
        if l == "--":
            data += [[torch.tensor(training_x, dtype=torch.double), torch.tensor(training_y, dtype=torch.double)]]

            training_x = []
            training_y = None
        elif l[:1] == "#":
            target = int(l[1:])
            training_y = [0.]*target + [1.] + [0.]*(9-target)
        else:
            local_row = []
            for v in l:
                local_row += [int(v)]
            
            training_x += local_row

    file.close()
    print("Data loaded")
    return data

training_data = load_training_data()
#print(training_data)
mu = 0.3


def output_all_case(model, data):
    print("Results-------")
    for item in data:
        x = item[0]
        print(model.output(x))

def fixed_training(model, epochs, data):
    for e in range(epochs):

        total_error = 0

        for item in data:
            x = item[0]
            t = item[1]

            model.train_once(x, t, mu)

            total_error += model.error

        print(total_error)

def train_until_pass(model, pass_rate, data):
    try:
        total_error = 10

        while total_error > pass_rate:
            total_error = 0

            for item in data:
                x = item[0]
                t = item[1]

                model.train_once(x, t, mu)

                total_error += model.error
            
            print(total_error)
    finally:
        output_all_case(perceptron, training_data)
        perceptron.save(txt_path)

train_until_pass(perceptron, 5, training_data)
#fixed_training(perceptron, 1000, training_data)


#output_all_case(perceptron, training_data)