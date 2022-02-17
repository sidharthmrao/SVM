import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Class with function to animate a graph

class Animate:
    def __init__(self, points_x, points_y, max_x, max_y):
        self.points_x = points_x
        self.points_y = points_y
        self.max_x = max_x
        self.max_y = max_y
        self.fig, self.ax = plt.subplots()
    def animate_frame(self):
        self.ax.clear()
        self.ax.set_xlim([0, self.max_x])
        self.ax.set_ylim([0, self.max_y])
        self.ax.plot(self.points_x, self.points_y)
        # Plot the line
        self.ax.plot([0, self.max_x], [self.b, self.max_x*self.m+self.b], color='red', linewidth=2)
    def start_animation(self, m, b):
        self.m = m
        self.b = b
        ani = FuncAnimation(self.fig, self.animate_frame, frames=1000, interval=500, repeat=False)
        plt.show()

