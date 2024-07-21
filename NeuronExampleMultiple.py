import numpy as np

#Constants
bigErr = 100000

#Helpers
def sigmoid(x): #Sigmoid function
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoidDerivative2(y):
    return y * (1 - y)

def error(y, t): #L2 Norm
    return 0.5 * np.power((t-y), 2)

def errorDerivative(y, t):
    return -(t-y)


class NeuronLayer:
    def __init__(self, w, b, f):
        self.w = w #weight list
        self.b = b #bias values
        self.f = f #non-linear function

        self.input = None
        self.output = None
        self.err = None
    
    #functions
    def updateWeights(self, err, u):
        self.err = err
        g = err * self.input

        if len(self.w) == len(g):
            self.w = (self.w + g) * u
            print(self.w)

    def getInputSize(self):
        return len(self.w)

    def getOutputRaw(self, x):
        self.input = x
        if len(x) == len(self.w[0]):
            if x.ndim == 1:
                return np.array([sum([x[j]*self.w[i][j] for j in range(len(x))]) + self.b[i] for i in range(len(self.w))])
            else:
                return np.array([sum([x[j][0]*self.w[i][j] for j in range(len(x))]) + self.b[i] for i in range(len(self.w))])
        else:
            return "Weight/Input Mismatch"

    def getOutput(self, x):
        self.input = x
        if len(x) == len(self.w[0]):
            self.output = self.f(self.getOutputRaw(x))
            return self.output
        else:
            return "Weight/Input Mismatch"


def makeDeepNeuronLayer(inputSize, outputSize):
    mag = np.sqrt(6) / np.sqrt(inputSize + outputSize)

    initW = np.array([np.array([(np.random.rand() * mag * 2 - mag) for i in range(inputSize)]) for j in range(outputSize)])
    initB = np.array([np.array([0]) for i in range(outputSize)])

    return NeuronLayer(initW, initB, sigmoid)


class TestModel:
    def __init__(self, inputSize, layers):

        outputSize = 1 #always 1 for now

        self.outputSize = outputSize
        self.inputSize = inputSize
        self.layers = layers

        self.layerList = [makeDeepNeuronLayer(inputSize, inputSize)]*(layers-1) + [makeDeepNeuronLayer(inputSize, outputSize)]
        self.output = None

        self.err = bigErr

    def run(self, x):
        self.output = x

        for layer in self.layerList:
            self.output = layer.getOutput(x)

        return self.output
    
    def trainOnce(self, x, t, u):
        y = self.run(x)
        ti = t

        for j in range(self.layers):
            i = -1 - j
            curLayer = self.layerList[i]

            #xi = curLayer.input
            yi = curLayer.output

            if i == -1:
                errTerms = (ti-yi) * sigmoidDerivative2(yi)
                curLayer.updateWeights(errTerms, u)
            else:
                nxtLayer = self.layerList[i+1]

                errTerms = sum(curLayer.w * nxtLayer.err) * sigmoidDerivative2(yi)
                curLayer.updateWeights(errTerms, u)


            #hidden



trainedAND = TestModel(2,3)
trainedXOR = TestModel(2,3)

def actualAND(x):
    a = x[0]
    b = x[1]
    if a > 0 and b > 0:
        return 1
    return 0

def actualXOR(x):
    a = x[0]
    b = x[1]
    if a == b:
        return 0
    return 1

for i in range(2000): #arbitrary amount of training
    randomInput = np.array([int(np.random.rand() * 2) for i in range(2)])

    trainedAND.trainOnce(randomInput, actualAND(randomInput), 0.8) #Those u values were cherry picked for best results
    trainedXOR.trainOnce(randomInput, actualXOR(randomInput), 0.8)

print("-------------AND Output")

print(trainedAND.run(np.array([0,0]))) #0, 0.5
print(trainedAND.run(np.array([0,1]))) #0, 0.5083178
print(trainedAND.run(np.array([1,0]))) #0, 0.50277067
print(trainedAND.run(np.array([1,1]))) #1, 0.51108744

print("-------------XOR Output")

print(trainedXOR.run(np.array([0,0]))) #0, 0.5
print(trainedXOR.run(np.array([0,1]))) #1, 0.4916822
print(trainedXOR.run(np.array([1,0]))) #1, 0.49722933
print(trainedXOR.run(np.array([1,1]))) #0, 0.48891256

#test = NeuronLayer(np.array([[3.7, 3.7], [2.9, 2.9]]), np.array([[-1.5], [-4.6]]), sigmoid)

#print(test.getOutputRaw(np.array([[1], [0]])))
#print(test.getOutput(np.array([[1], [0]])))

