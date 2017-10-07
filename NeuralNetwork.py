#This is basically the neural network for the project
import numpy as np
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('XORdataset.csv')
datasetDimensions = dataset.shape
X = dataset.iloc[0:, :-1].values
y = dataset.iloc[:, -1].values
y = np.reshape(y, (4,1))



#Sigmoid Function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

#Derivative of sigmoid function
def sigmoidPrime(z):
    return np.exp(-z)/((1 + np.exp(-z))**2)



#The Neural Network Class
class NeuralNetwork(object):
#Initialise the layer sizes
    numberOfLayers = 3
    inputLayerSize = 2
    hiddenLayerSize = 3
    outputLayerSize = 1

#CREATE AND INTIALIZE THE WEIGHT MATRICES W1 AND W2 TO NEAR ZERO VALUES
    def __init__(self):
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        self.W1b = np.random.randn(1, self.hiddenLayerSize)
        self.W2b = np.random.randn(1, self.outputLayerSize)
        self.debug = 1

#INITIALIZE THE VALUES OF Z2, A2, Z3, A3(OUTPUT)
    def initialiseAZ(self, X):
        self.Z2 = np.dot(X, self.W1)
        self.Z2 = self.Z2 + self.W1b
        self.A2 = sigmoid(self.Z2)
        self.Z3 = np.dot(self.A2, self.W2)
        self.Z3 = self.Z3 + self.W2b
        self.output = sigmoid(self.Z3)

#FINDING THE COST
    def costFunction(self):
        return (1/2)*sum((self.output - y)**2)

#DEFINE THE FORWARD FEED
    def forward(self):
        self.initialiseAZ(X)
        if(self.debug==1):
            print('\nZ2 = ', self.Z2, '\nA2 = ', self.A2, '\nZ3 = ', self.Z3, '\nOuput = ', self.output)
            self.debug = 0
        #CALCULATE THE DERIVATIVES OF COST WITH RESPECT TO W1 AND W2
        delta3 = np.multiply(sigmoidPrime(self.Z3), (self.output - y))
        dJdW2 = np.dot(self.A2.T, delta3)
        delta2 = (np.dot(delta3, self.W2.T))*sigmoidPrime(self.Z2)
        dJdW1 = np.dot(X.T, delta2)

        print('\ndJdW1 = ', dJdW1, '\ndJdW2 = ', dJdW2)
        scalar = 1
        self.W1 = self.W1 - scalar*dJdW1
        self.W2 = self.W2 - scalar*dJdW2
        self.W1b = self.W1b - scalar*sum(delta2)
        self.W2b = self.W2b - scalar*sum(delta3)
        return self.W1, self.W2, self.W1b, self.W2b


#Main code
myNeuralNetwork = NeuralNetwork()
initialDJDW1, initialDJDW2, bw1, bw2 = myNeuralNetwork.forward()
initialCost = myNeuralNetwork.costFunction()


for i in range(1,1000):
    weight1, weight2, weightBias1, weightBias2 = myNeuralNetwork.forward()
    cost = myNeuralNetwork.costFunction()
    print('Cost =', cost)

print('\ninitialDJDW1 = ', initialDJDW1, '\ninitialDJDW2', initialDJDW2)
print('\ninitialCost = ', initialCost)

#Till here we have the values of optimised weights
#Here the input should include the bias unit 1 i.e the input should be like 1 0 0
print('Enter the values of input: ')
x = input()
x = x.split(' ')
x = np.array(x, dtype='int64')
x = np.reshape(x, (1,2))

testZ2 = np.dot(x, weight1)
testZ2 = testZ2 + weightBias1
testA2 = sigmoid(testZ2)
testZ3 = np.dot(testA2, weight2)
testZ3 = testZ3 + weightBias2
testOutput = sigmoid(testZ3)

print('\nOUTPUT = ', testOutput)
