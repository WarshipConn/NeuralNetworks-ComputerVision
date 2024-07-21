import numpy as np
import torch

class ExampleNet(torch.nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.layer1 = torch.nn.Linear(2,2)
        self.layer2 = torch.nn.Linear(2,1)
        self.layer1.weight = torch.nn.Parameter(torch.tensor([[3.,4.],[2.,3.]]))
        self.layer1.bias = torch.nn.Parameter(torch.tensor([-2.,-4.]))
        self.layer2.weight = torch.nn.Parameter(torch.tensor([[5.,-5.]]))
        self.layer2.bias = torch.nn.Parameter(torch.tensor([-2.]))

    def forward(self, x):
        s = self.layer1(x)
        h = torch.nn.Sigmoid()(s)
        z = self.layer2(h)
        y = torch.nn.Sigmoid()(z)
        return y

#XOR data
x = torch.tensor([ [0.,0.], [1.,0.], [0.,1.], [1.,1.] ])
t = torch.tensor([ 0., 1., 1., 0. ])
mu = 0.1

net = ExampleNet()
optimizer = torch.optim.SGD(net.parameters(), lr=mu)

#XOR data
x = torch.tensor([ [0.,0.], [1.,0.], [0.,1.], [1.,1.] ])
t = torch.tensor([ 0., 1., 1., 0. ])

def output_all_case(model, data):
    print("Results-------")
    for x in data:
        print(model.forward(x))

def train(epochs):
    for iteration in range(epochs):
        optimizer.zero_grad()
        out = net.forward( x )
        error = 1/2 * (t - out) ** 2
        mean_error = error.mean()
        print("error: ",mean_error.data)
        mean_error.backward()
        optimizer.step()


output_all_case(net, x)

train(10000)

output_all_case(net, x)
