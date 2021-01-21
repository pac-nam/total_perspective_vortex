import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class AnimatedScatter(object):
    def __init__(self, numpoints=64):
        self.numpoints = numpoints
        self.stream = self.data_stream()

        # Setup the figure and axes...
        # self.img = plt.imread("eeg_schema.png")
        self.fig, self.ax = plt.subplots(figsize=(7,7))
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1, 
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        x, y, s, c = next(self.stream).T
        # self.ax.imshow(img)
        self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=0, vmax=1,
                                    cmap="jet", edgecolor="k")
        self.ax.axis([-11, 11, -11, 11])
        return self.scat,

    def data_stream(self):
        xy = [
            [-5.8,  2.3], [-3.9,  2.2], [-2.0,  2.1], [ 0.0,  2.0], [ 2.0,  2.1], [ 3.9,  2.2], [ 5.8,  2.3], [-6.0,  0.0],
            [-4.0,  0.0], [-2.0,  0.0], [ 0.0,  0.0], [ 2.0,  0.0], [ 4.0,  0.0], [ 6.0,  0.0], [-5.8, -2.3], [-3.9, -2.2],
            [-2.0, -2.1], [ 0.0, -2.0], [ 2.0, -2.1], [ 3.9, -2.2], [ 5.8, -2.3], [-2.8,  7.8], [ 0.0,  8.0], [ 2.8,  7.8],
            [-5.0,  6.4], [-3.1,  6.1], [ 0.0,  6.0], [ 3.1,  6.1], [ 5.0,  6.4], [-6.8,  4.7], [-5.3,  4.4], [-3.7,  4.2],
            [-1.7,  4.1], [ 0.0,  4.0], [ 1.7,  4.1], [ 3.7,  4.2], [ 5.3,  4.4], [ 6.8,  4.7], [-7.7,  2.6], [ 7.7,  2.6],
            [-8.0,  0.0], [ 8.0,  0.0], [ -10,  0.0], [10.0,  0.0], [-7.7, -2.8], [ 7.7, -2.8], [-6.7, -4.8], [-5.2, -4.4],
            [-3.6, -4.2], [-1.7, -4.1], [ 0.0, -4.0], [ 1.7, -4.1], [ 3.6, -4.2], [ 5.2, -4.4], [ 6.7, -4.8], [-5.0, -6.4],
            [-3.2, -5.8], [ 0.0, -6.0], [3.2, -5.8], [ 5.0, -6.4], [-2.6, -7.5], [ 0.0, -8.0], [ 2.6, -7.5], [ 0.0,  -10]
        ]
        s = (np.ones(self.numpoints)) + 0.4
        c = np.zeros(self.numpoints)
        # print((np.random.random((self.numpoints, 2)) - 0))
        while True:
            xy += np.zeros((self.numpoints, 2))
            c -= (np.zeros(self.numpoints)) - 0.02
            yield np.c_[xy[:,0], xy[:,1], s, c]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)
        # self.ax.imshow(self.img)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        # Set colors..
        self.scat.set_array(data[:, 3])
        return self.scat,

if __name__ == '__main__':
    a = AnimatedScatter()
    plt.show()