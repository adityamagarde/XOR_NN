#This is basically the neural network for the project
import numpy as np
import pandas as pd

dataset = pd.read_csv(r'/home/adityamagarde/Documents/Projects/Machine Learning Basics/XORdataset.csv')
datasetDimensions = dataset.shape

inputSet = dataset.iloc[0:, :-1].values
rowsOnes = np.ones((datasetDimensions[0], 1), dtype='int64')
X = np.concatenate((inputSet, rowsOnes), axis=1)
X = np.roll(X, 1, axis=1)

outputSet = dataset.iloc[:, -1].values
y = np.reshape(outputSet, (datasetDimensions[0],1))


initialDJDW1 = 1
initialDJDW2 = 1

#SIGMOID FUNCTION
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoidPrime(z):
    return np.exp(-z)/((1 + np.exp(-z))**2)

class NeuralNetwork(object):
#INTIALIZE THE LAYER SIZES
    numberOfLayers = 3
    inputLayerSize = 2
    hiddenLayerSize = 3
    outputLayerSize = 1

#CREATE AND INTIALIZE THE WEIGHT MATRICES W1 AND W2 TO NEAR ZERO VALUES
    def __init__(self):
        self.W1 = np.random.randn(self.inputLayerSize + 1, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize + 1, self.outputLayerSize)

#INITIALIZE THE VALUES OF Z2, A2, Z3, A3(OUTPUT)
    def initialiseAZ(self, X):
        self.Z2 = np.dot(X, self.W1)
        self.A2 = sigmoid(self.Z2)
        
        shapeOfX = X.shape
        rowsOnes = np.ones((shapeOfX[0],1), dtype='int64')
        #adding bias to A2
                
        self.A2 = np.concatenate((self.A2, rowsOnes), axis=1)
        self.A2 = np.roll(self.A2, 1, axis=1)
        
        self.Z3 = np.dot(self.A2, self.W2)
        self.output = sigmoid(self.Z3)

#FINDING THE COST+
    def costFunction(self):
        return (1/2)*sum((self.output - y)**2)

#DEFINE THE FORWARD FEED
    def forward(self):
        self.initialiseAZ(X)
    #    cost = self.costFunction()
    #CALCULATE THE DERIVATIVES OF COST WITH RESPECT TO W1 AND W2
        delta3 = np.multiply(sigmoidPrime(self.Z3), (self.output - y))
        dJdW2 = np.dot(self.A2.T, delta3)
        delta2 = np.dot((np.dot(delta3, self.W2.T)) , sigmoidPrime(self.Z2))
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2

    def backwards(self):
        dJdW1, dJdW2 = self.forward()
        print('dJdW1 = ', dJdW1, 'dJdW2 = ', dJdW2)
        scalar = 1
        self.W1 = self.W1 - scalar*dJdW1
        self.W2 = self.W2 - scalar*dJdW2
        return self.W1, self.W2

myNeuralNetwork = NeuralNetwork()
#updating weights is required
initialDJDW1, initialDJDW2 = myNeuralNetwork.forward()
initialCost = myNeuralNetwork.costFunction()


for i in range(1,1000):
    weight1, weight2 = myNeuralNetwork.backwards()
    cost = myNeuralNetwork.costFunction()
    print('W1 = ', weight1, 'W2 = ', weight2, 'Cost =', cost)
    
print('initialDJDW1 = ', initialDJDW1, '\ninitialDJDW2', initialDJDW2)
print('initialCost = ', initialCost)
#Now I have the optimised weights: weight1, weight2


#Here the input should include the bias unit 1 i.e the input should be like 1 0 0
print('Enter the values of input: ')
x = raw_input()
x = x.split(' ')
x = np.array(x, dtype='int64')
x = np.reshape(x, (1,3))

testZ2 = np.dot(x, weight1)
testA2 = sigmoid(testZ2)

testShapeOfX = x.shape
testRowsOnes = np.ones((testShapeOfX[0],1), dtype='int64')
testA2 = np.concatenate((testRowsOnes, testA2), axis=1)

testZ3 = np.dot(testA2, weight2)
testOutput = sigmoid(testZ3)

print('\nOUTPUT = ', testOutput)