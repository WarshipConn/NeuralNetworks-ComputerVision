import numpy as np

#Constants
bigErr = 100000

#Helpers
def sigmoid(x): #Sigmoid function
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def error(y, t): #L2 Norm
    return 0.5 * np.power((t-y), 2)

def errorDerivative(y, t):
    return -(t-y)

class NeuronLayerSingle:
    def __init__(self, w, b, f):
        self.w = w #weight list
        self.b = b #bias value
        self.f = f #non-linear function
    
    #functions
    def updateWeights(self, g, u):
        if len(self.w) == len(g):
            self.w = (self.w + g) * u
            print(self.w)

    def getInputSize(self):
        return len(self.w)

    def getOutputRaw(self, x):
        if len(x) == len(self.w):
            return sum(x*self.w) + self.b
        else:
            return "Weight/Input Mismatch"

    def getOutput(self, x):
        if len(x) == len(self.w):
            return self.f(self.getOutputRaw(x))
        else:
            return "Weight/Input Mismatch"


class TestModelSingle: #Single Layer Only
    def __init__(self, inputSize):

        mag = 1 / np.sqrt(inputSize) #shallow weight initialization

        self.layer = NeuronLayerSingle(np.array([(np.random.rand() * mag * 2 - mag) for i in range(inputSize)]), 0., sigmoid) #Neuron layer, init values [-1/sqrt(x), 1/sqrt(x)]
        self.output = None

        self.err = bigErr

    def run(self, x):
        self.output = self.layer.getOutput(x)
        return self.output
    
    def trainOnce(self, x, t, u):
        y = self.run(x)
        s = self.layer.getOutputRaw(x)
        err = errorDerivative(y, t)
        g = err * sigmoidDerivative(s) * x

        errResult = "is equal"
        if self.err > err:
            errResult ="is lower"
        elif self.err < err:
            errResult ="is higher"
        
        self.err = err

        #print(f"Input: {x} | Generated Answer: {y} | Actual Answer: {t} (Error rate is {err}, {errResult})")

        self.layer.updateWeights(g, u)


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

trainedAND = TestModelSingle(2)
trainedXOR = TestModelSingle(2)


andOriginalWeight = trainedAND.layer.w
xorOriginalWeight = trainedXOR.layer.w

for i in range(250): #arbitrary amount of training
    randomInput = np.array([int(np.random.rand() * 2) for i in range(2)])

    trainedAND.trainOnce(randomInput, actualAND(randomInput), 0.97) #Those u values were cherry picked for best results
    trainedXOR.trainOnce(randomInput, actualXOR(randomInput), 0.97)

    '''fixedInput = np.array([0, 0])

    trainedAND.trainOnce(fixedInput, actualAND(fixedInput), 0.1) #Those u values were cherry picked for best results
    trainedXOR.trainOnce(fixedInput, actualXOR(fixedInput), 0.1)

    fixedInput = np.array([0, 1])

    trainedAND.trainOnce(fixedInput, actualAND(fixedInput), 0.1)
    trainedXOR.trainOnce(fixedInput, actualXOR(fixedInput), 0.1)

    fixedInput = np.array([1, 0])

    trainedAND.trainOnce(fixedInput, actualAND(fixedInput), 0.1) 
    trainedXOR.trainOnce(fixedInput, actualXOR(fixedInput), 0.1)

    fixedInput = np.array([1, 1])

    trainedAND.trainOnce(fixedInput, actualAND(fixedInput), 0.1)
    trainedXOR.trainOnce(fixedInput, actualXOR(fixedInput), 0.1)'''

#Note: u valueswere cherry picked for better results

#Another Note: Bias is always set to 0, so when input is [0, 0], the result would always be 0.5, since weights won't affect output

print("-------------AND Output")

print(trainedAND.run(np.array([0,0]))) #0, 0.5
print(trainedAND.run(np.array([0,1]))) #0, 0.8424169432809103
print(trainedAND.run(np.array([1,0]))) #0, 0.6769878875770685
print(trainedAND.run(np.array([1,1]))) #1, 0.9180607893800921

print("-------------XOR Output")

print(trainedXOR.run(np.array([0,0]))) #0, 0.5
print(trainedXOR.run(np.array([0,1]))) #1, 0.10408986683883949
print(trainedXOR.run(np.array([1,0]))) #1, 0.2770619324755566
print(trainedXOR.run(np.array([1,1]))) #0, 0.042628520033774785

#Random inputs (Multiple neurons)

'''
x = np.array((1, 2))
w = np.array(((1,2), (3,4), (5,6))) #3 neurons, 2 input & weight pairs each
b = np.array((1, 2, 3))

def f(x): #Sigmoid function
    return 1 / (1 + np.exp(-x))

def neurons(x, w, b, f):
    if w.shape[0] == len(b):
        if w.shape[1] == len(x):

            xw = np.array([sum(i) for i in x*w])
            result = [0] * w.shape[0]
            
            for i in range(w.shape[0]):
                result[i] = f(xw[i] + b[i])

            return result
        else:
            return "Mismatch between inputs and weights"
    else:
        return "Missing weights or bias for some neurons"

print(neurons(x, w, b, f))
# sum(X*W) + B: [6, 13, 20]
#output: [0.9975273768433653, 0.999997739675702, 0.9999999979388463]

'''
