import numpy as np
from random import uniform
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# TODO Add Least Squares Regression to this

class gradient_descent:
    def __init__(self, x: np.array, y, lr=.12, epochs=3000):
        self.x = x
        self.y = y
        self.n = len(x)
        self.lr = lr
        self.epochs = epochs
        self.m = uniform(0, 1)
        self.b = uniform(0, 1)
        self.m = 5
        self.b = 5
        self.logs = []
        self.mselog = []

    def funct(self, x):
        return self.m*x + self.b

    def MSE(self, x, y, ypred):
        return sum((y[i]-ypred[i])**2 for i in range(len(x)))/len(x) 
    def MSE_M(self, x, y: list, ypred: list):
        return sum(-2*x[i]*(y[i]-ypred[i]) for i in range(len(x)))/len(x)
    def MSE_B(self, x, y: list, ypred: list):
        return sum(-2*(y[i]-ypred[i]) for i in range(len(x)))/len(x)
    
    def gradient_descent(self):
        for i in range(self.epochs):
            predicted_y = [self.funct(x) for x in self.x]
            self.m -= self.lr * self.MSE_M(self.x, self.y, predicted_y)
            self.b -= self.lr * self.MSE_B(self.x, self.y, predicted_y)

            self.logs.append((self.m, self.b))
            self.mselog.append(self.MSE(self.x, self.y, predicted_y))
        return self.m, self.b, self.logs, self.mselog, self.mselog[-1]

X = np.array(sorted(list(range(5))*20)) + np.random.normal(size=100, scale=0.5)
y = np.array(sorted(list(range(5))*20)) + np.random.normal(size=100, scale=0.25)

pointsx = X
pointsy = y
print(pointsx)
print(pointsy)

GD = gradient_descent(pointsx, pointsy)
m, b, logs, mselog, loss = GD.gradient_descent()
print(round(m, 2), round(b, 2), round(loss, 2))


def plotmb(m, b, ax, alpha=1, color='red', linewidth=.2):
  ax.plot([-1000,1000], [-1000*m+b, 1000*m+b], color=color, alpha=alpha, linewidth=linewidth)

#for i in logs:
  #plotmb(i[0], i[1], .9)
"""
plt.plot(pointsx, pointsy, 'bo')
plotmb(logs[-1][0], logs[-1][1], color='red', linewidth=2)
plt.fill_between([-1000,1000], [-1000*logs[0][0]+logs[0][1], 1000*logs[0][0]+logs[0][1]], [-1000*logs[-1][0]+logs[-1][1], 1000*logs[-1][0]+logs[-1][1]], alpha=.7, color="red")
plt.axis([-2, 6, -2, 6])
plt.show()
"""

fig, ax = plt.subplots()


def animate_frame(i):
    ax.clear()
    ax.set_xlim([-2.5, 7.5])
    ax.set_ylim([-2.5, 7.5])
    ax.plot(pointsx, pointsy, 'bo')
    for line in logs[:i]:
        plotmb(line[0], line[1], ax, alpha=.9, color='red', linewidth=.3)
    plotmb(logs[i][0], logs[i][1], ax, color='red', linewidth=2)
    # Plot the line
    #ax.plot([0, 5], [b, 5*m+b], color='red', linewidth=2)
    print(i)

ani = FuncAnimation(fig, animate_frame, frames=3000, interval=50, repeat=False)
plt.show()