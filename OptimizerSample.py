import numpy as np
import torch

class NeuronLayer():
    def __init__(self, w, b):
        self.w = w
        self.b = b

def make_neuron_layer(input_size, output_size):
    mag = np.sqrt(6) / np.sqrt(input_size + output_size)

    w = [[(np.random.rand() * mag * 2 - mag) for i in range(input_size)] for j in range(output_size)]
    #w = [[1]*input_size]*output_sizemag
    b = [mag - np.random.rand() * 2 * mag]*output_size

    layer = torch.nn.Linear(input_size, output_size)
    layer.weight =  torch.nn.Parameter(torch.tensor(w))
    layer.bias =  torch.nn.Parameter(torch.tensor(b))

    return layer

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, layers):
        output_size = 1 #always 1 for now

        self.output_size = output_size
        self.input_size = input_size
        self.layers = layers

        self.error = 0

        layer1 = torch.nn.Linear(2,2)
        layer1.weight = torch.nn.Parameter(torch.tensor([[3,4],[2,3]], dtype=torch.double))
        layer1.bias = torch.nn.Parameter(torch.tensor([-2,-4], dtype=torch.double))
        layer2 = torch.nn.Linear(2,1)
        layer2.weight = torch.nn.Parameter(torch.tensor([[5,-5]], dtype=torch.double))
        layer2.bias = torch.nn.Parameter(torch.tensor([-2], dtype=torch.double))
        self.layer_list = [layer1, layer2]
        
        #self.layer_list = [make_neuron_layer(input_size, input_size)]*(layers-1) + [make_neuron_layer(input_size, output_size)]
    
    def forward(self, x):
        y = x
        for layer in self.layer_list:
            s = layer(x)
            y = torch.nn.Sigmoid()(s)
        return y
    
    def train_once(self, x, t, mu):
        y = self.forward(x)

        error = 1/2 * (t - y) ** 2

        #print(error.data.numpy()[0])
        self.error = error.data.numpy()[0]

        error.backward()

        for layer in self.layer_list:
            layer.w.data = layer.w - mu * layer.w.grad.data
            layer.b.data = layer.b - mu * layer.b.grad.data

            layer.w.grad.data.zero_()
            layer.b.grad.data.zero_()
        
    def output(self, x):
        y = self.forward(x)
        return y.detach().numpy()



#XOR data
training_data =[ 
    [ torch.tensor([0.,0.], dtype=torch.double), torch.tensor([0.]) ],
    [ torch.tensor([1.,0.], dtype=torch.double), torch.tensor([1.]) ],
    [ torch.tensor([0.,1.], dtype=torch.double), torch.tensor([1.]) ],
    [ torch.tensor([1.,1.], dtype=torch.double), torch.tensor([0.]) ]
]
mu = 0.01

XOR_model = NeuralNetwork(2,2)

optimizer =  torch.optim.SGD(XOR_model.parameters(), lr=0.1)

def train_with_optimizer(model, epochs, data):
    optimizer =  torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):

        for item in data:
            x = item[0]
            t = item[1]

            optimizer.zero_grad()
            out = model.forward(x)
            error = 1/2 * (t - out) ** 2
            mean_error = error.mean()
            print("error: ",mean_error.data)
            mean_error.backward()
            optimizer.step()


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

        #print(total_error)

def random_training_equal_distribution(model, epochs, data):
    for e in range(epochs):
        training_set = data.copy()
        n = len(training_set)
        for i in range(n-1, 0, -1):
            j = int(np.random.rand() * (i+1))
            temp = training_set[i]
            training_set[i] = training_set[j]
            training_set[j] = temp

        for item in training_set:
            x = item[0]
            t = item[1]

            model.train_once(x, t, mu)

def random_training_unequal_distribution(model, epochs, data):
    for e in range(epochs):
        for i in range(len(data)):
            j = int(np.random.rand() * len(data))

            x = data[j][0]
            t = data[j][1]

            model.train_once(x, t, mu)

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
        output_all_case(XOR_model, training_data)

output_all_case(XOR_model, training_data)

#fixed_training(XOR_model, 1000, training_data)
#train_until_pass(XOR_model1, 0.1, training_data)
#random_training_equal_distribution(XOR_model2, 1000, training_data)
#random_training_unequal_distribution(XOR_model3, 1000, training_data)
train_with_optimizer(XOR_model, 1000, training_data)

output_all_case(XOR_model, training_data)
print("-------")

'''for item in training_data:
    x = item[0]
    print(XOR_model2.output(x))
print("-------")
for item in training_data:
    x = item[0]
    print(XOR_model3.output(x))
print("-------")'''

'''W = torch.tensor([[3,4],[2,3]], requires_grad=True, dtype=float)
b = torch.tensor([-2,-4], requires_grad=True, dtype=float)
W2 = torch.tensor([5,-5], requires_grad=True, dtype=float)
b2 = torch.tensor([-2], requires_grad=True, dtype=float)

for epoch in range(1000):
    total_error = 0
    for item in training_data:
        x = item[0]
        t = item[1]

        s = W.mv(x) + b
        h = torch.nn.Sigmoid()(s)
        z = torch.dot(W2, h) + b2
        y = torch.nn.Sigmoid()(z)

        error = 1/2 * (t - y) ** 2
        total_error = total_error + error

    total_error.backward()

    W.data = W - mu * W.grad.data
    b.data = b - mu * b.grad.data
    W2.data = W2 - mu * W2.grad.data
    b2.data = b2 - mu * b2.grad.data

    W.grad.data.zero_()
    b.grad.data.zero_()
    W2.grad.data.zero_()
    b2.grad.data.zero_()


for item in training_data:
    x = item[0]

    s = W.mv(x) + b
    h = torch.nn.Sigmoid()(s)
    z = torch.dot(W2, h) + b2
    y = torch.nn.Sigmoid()(z)

    print(y.detach().numpy())'''