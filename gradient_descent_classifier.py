import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

read = pd.read_csv("data.csv")

vars = []
for i in read.columns[0:-1]:
    vars.append(list(read[i]))
labels = list(read[read.columns[-1]])
print(vars)
print(labels)

# Create a random number for weight for each variable
weights = [np.random.uniform(0,1) for i in range(len(vars))]
print(weights)

class gradient_descent:
    def __init__(self, weights, vars, labels, lr=.01, epochs=1000):
        self.vars = vars
        self.weights = weights
        self.labels = labels
        self.lr = lr
        self.epochs = epochs
        self.b = np.random.uniform(0, 1)
        self.logs = []
        self.mselog = []

    def funct(self, index):
        total = 0
        for i, var in enumerate(vars):
            total+=var[index]*self.weights[i]
        total+=self.b
        return total

    def label_prediction(self, index):
        pass
    def MSE(self, predicted_y):
        return sum((self.labels[index]-predicted_y[index])**2 for index in range(len(self.labels)))/len(self.labels)
    def MSE_M(self, predicted_y, m_index):
        return sum(-2*vars[m_index][index]*(self.labels[index]-predicted_y[index])**2 for index in range(len(self.labels)))/len(self.labels)
    def MSE_B(self, predicted_y):
        return sum(-2*(self.labels[index]-predicted_y[index])**2 for index in range(len(self.labels)))/len(self.labels)
    
    def gradient_descent(self):
        for i in range(self.epochs):
            predicted_y = []
            for index in range(len(vars[0])):
                predicted_y.append(self.funct(index))
            for m in range(len(weights)):
                weights[m] -= self.lr*self.MSE_M(predicted_y, m)
            self.b -= self.lr*self.MSE_B(predicted_y)
            self.logs.append((self.weights, self.b))
            print(self.MSE(predicted_y))
            self.mselog.append(self.MSE(predicted_y))
        return self.weights, self.b, self.logs, self.mselog, self.mselog[-1]

X = np.array(sorted(list(range(5))*20)) + np.random.normal(size=100, scale=0.5)
y = np.array(sorted(list(range(5))*20)) + np.random.normal(size=100, scale=0.25)

pointsx = X
pointsy = y
#print(pointsx)
#print(pointsy)

GD = gradient_descent(weights, vars, labels)
m, b, logs, mselog, loss = GD.gradient_descent()
print("Weights: ", m)
print("Bias: ", b)
print("Loss: ", loss)

