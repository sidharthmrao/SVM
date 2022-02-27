import numpy as np
from random import uniform
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import imageio
import os

# TODO Add Least Squares Regression to this

class gradient_descent:
    def __init__(self, x: np.array, y: np.array, z: list, lr=.01, epochs=1500):
        self.x = x
        self.y = y
        self.z = z
        self.n = len(x)
        self.lr = lr
        self.epochs = epochs
        self.m1 = uniform(0, 1)
        self.m2 = uniform(0, 1)
        self.b = uniform(0, 1)
        self.m1 = 5
        self.m2 = 5
        self.b = 5
        self.logs = []
        self.mselog = []

    def funct(self, i):
        return self.m1*self.x[i] + self.m2*self.y[i] + self.b

    def MSE(self, z, zpred):
        return sum((z[i]-zpred[i])**2 for i in range(self.n))/self.n 
    def MSE_M1(self, x, z: list, zpred: list):
        return sum(-2*x[i]*(z[i]-zpred[i]) for i in range(self.n))/self.n 
    def MSE_M2(self, y, z: list, zpred: list):
        return sum(-2*y[i]*(z[i]-zpred[i]) for i in range(self.n))/self.n 
    def MSE_B(self, z: list, zpred: list):
        return sum(-2*(z[i]-zpred[i]) for i in range(self.n))/self.n 
    
    def gradient_descent(self):
        for i in range(self.epochs):
            
            # BATCH
            predicted_z = [self.funct(i) for i in range(self.n)]
            #print(predicted_z)
            self.m1 -= self.lr * self.MSE_M1(self.x, self.z, predicted_z)
            self.m2 -= self.lr * self.MSE_M2(self.y, self.z, predicted_z)
            self.b -= self.lr * self.MSE_B(self.z, predicted_z)

            self.logs.append((self.m1, self.m2, self.b))
            self.mselog.append(self.MSE(self.z, predicted_z))
            
            #STOCHASTIC
            # sample_size = .1

            # selected_x_index = np.random.choice(self.n, int(self.n*sample_size), replace=True)
            # selected_x = [self.x[i] for i in selected_x_index]
            # selected_y = [self.y[i] for i in selected_x_index]
            # predicted_y = [self.funct(x) for x in selected_x]
            # self.m -= self.lr * self.MSE_M(selected_x, selected_y, predicted_y)
            # self.b -= self.lr * self.MSE_B(selected_x, selected_y, predicted_y)
            # self.logs.append((self.m, self.b))
            # self.mselog.append(self.MSE(selected_x, selected_y, predicted_y))
            

            
        return self.m1, self.m2, self.b, self.logs, self.mselog, self.mselog[-1]

# BATCH
# x = np.array([1, 7, 11, 3, 2, 5, 5, 5, 0, -2, -4, 2, 3, 4.5, 2.3, 1.2])
# y = np.array([-2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1])
# z = np.array([7.2, 7.3, 7.4, 8.1, 8.5, 8.9, 9.1, 9.5, 9.9, 2, 3, 4, 10.2, 10.5, 11.5, 12])
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
z = np.array([1,2,3,4,5,6,7,8,9,10])

# STOCHASTIC
# x = np.array(sorted(list(range(5))*20)) + np.random.normal(size=100, scale=2)
# y = np.array(sorted(list(range(5))*20)) + np.random.normal(size=100, scale=.5)

print(x)
print(y)
print(z)

GD = gradient_descent(x, y, z)
m1, m2, b, logs, mselog, loss = GD.gradient_descent()
print(round(m1, 2), round(m2, 2), round(b, 2), round(loss, 2))


def plotmb(m1, m2, b, ax, alpha=1, color='red', linewidth=.2):
    x = np.arange(-2.5, 12, .1)
    y = np.arange(-2.5, 12, .1)
    z = [m1*x[i] + m2*y[i] + b for i in range(len(x))]
    ax.plot3D(x, y, z, color=color, alpha=alpha, linewidth=linewidth)

#for i in logs:
  #plotmb(i[0], i[1], .9)
"""
plt.plot(pointsx, pointsy, 'bo')
plotmb(logs[-1][0], logs[-1][1], color='red', linewidth=2)
plt.fill_between([-1000,1000], [-1000*logs[0][0]+logs[0][1], 1000*logs[0][0]+logs[0][1]], [-1000*logs[-1][0]+logs[-1][1], 1000*logs[-1][0]+logs[-1][1]], alpha=.7, color="red")
plt.axis([-2, 6, -2, 6])
plt.show()
"""

fig = plt.figure()
ax = plt.axes(projection='3d')





# def animate_frame(i):

#     if i%30==0:
#         ax.clear()
#         ax.scatter3D(x, y, z, c='r', marker='o')

#         ax.set_xlim([-2.5, 12])
#         ax.set_ylim([-2.5, 12])
#         ax.set_zlim([-2.5, 12])

#         j = 0

#         for line in logs[:i]:
#             j += 1
#             if j%30==0:
#                 plotmb(line[0], line[1], line[2], ax, alpha=.9, color='red', linewidth=.2)
#         plotmb(logs[i][0], logs[i][1], logs[i][2], ax, color='red', linewidth=2)
#         ax.set_xlabel(f"M: {round(logs[i][0], 2)} B: {round(logs[i][1], 2)} Loss: {round(mselog[i], 2)}")

#         print(i)

# ani = FuncAnimation(fig, animate_frame, frames=300, interval=1, repeat=False)
# plt.show()



images = []
for i in range(1500):
    if i%1499 == 0:
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(x, y, z, c='r', marker='o')
        ax.set_xlim([-2.5, 12])
        ax.set_ylim([-2.5, 12])
        ax.set_zlim([-2.5, 12])

        j=0
        for line in logs[:i]:
            j += 1
            if j%15==0:
                plotmb(line[0], line[1], line[2], ax, alpha=.9, color='red', linewidth=.2)
        plotmb(logs[i][0], logs[i][1], logs[i][2], ax, color='red', linewidth=2)

        ax.set_xlabel(f"MX: {round(logs[i][0], 2)} MY: {round(logs[i][1], 2)} B: {round(logs[i][2], 2)} Loss: {round(mselog[i], 2)} Iteration = {i}")
        print(i)

        fname = 'tmp/tmp%03d.png' % i
        fig.savefig(fname)
        images.append(Image.open(fname))
imageio.mimsave('animation3dshortpart2.gif', images)

print("Complete")