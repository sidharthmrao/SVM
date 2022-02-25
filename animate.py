import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
z = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

def animate_frame(i):
    ax.clear()
    ax.scatter3D(x, y, z, c='r', marker='o')

ani = FuncAnimation(fig, animate_frame, frames=30, interval=1, repeat=False)

plt.show()

