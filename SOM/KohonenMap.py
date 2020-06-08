import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import distance
import matplotlib

matplotlib.use("TkAgg")
import GeneratePoints


class SelfOrganizingMap(object):
    def __init__(self, number_of_neurons, input_data_file, radius, alpha, gaussian):
        self.radius = radius
        self.max_radius = radius
        self.min_radius = 0000.1
        self.all_steps = 0
        self.alpha = alpha
        self.max_alpha = alpha
        self.min_alpha = 0000.1
        self.p_min = 0.75
        # np.random.seed(20)
        self.gaussian = gaussian
        self.number_of_neurons = number_of_neurons
        self.input_data = self.file_input(input_data_file)
        self.neuron_weights = np.random.normal(np.mean(self.input_data), np.std(self.input_data),
                                               size=(self.number_of_neurons, len(self.input_data[0])))
        self.distance = []
        self.winner = -1
        self.neighborhood = []
        self.winner_distance = []
        self.testData = self.file_input("testData.txt")
        self.error = []
        self.potential = np.ones(self.number_of_neurons)
        self.activation = np.ones(self.number_of_neurons)
        self.animation_plots = []

    def file_input(self, file_name):
        input_arr = []
        with open(file_name, "r") as f:
            data = csv.reader(f, delimiter=',')
            for row in data:
                tmp_arr = []
                for i in row:
                    tmp_arr.append(float(i))
                input_arr.append(tmp_arr)
        return np.asarray(input_arr)

    def calculate_distance(self, inp, for_calculate, distance_calculation):
        for i in for_calculate:
            distance_calculation.append(distance.euclidean(i, inp))

    def find_winner(self):
        sorted_distance = np.argsort(np.asarray(self.distance))
        for i in sorted_distance:
            if self.activation[i] != 0:
                self.winner = i
                break

    def kohonen_neighborhood(self):
        if self.gaussian == 1:
            for i in self.winner_distance:
                self.neighborhood.append(math.exp(-1 * (i ** 2) / (2 * self.radius ** 2)))
        else:
            for i in self.winner_distance:
                if i <= self.radius:
                    self.neighborhood.append(1)
                else:
                    self.neighborhood.append(0)

    def clear_lists(self, step):
        self.neuron_activation()
        self.distance.clear()
        self.neighborhood.clear()
        self.winner_distance.clear()
        self.radius = self.max_radius * (self.min_radius / self.max_radius) ** (step / self.all_steps)
        self.alpha = self.max_alpha * (self.min_alpha / self.max_alpha) ** (step / self.all_steps)

    def update_weights(self, inp):
        for i in range(len(self.neuron_weights)):
            if self.activation[i] == 1:
                self.neuron_weights[i] = self.neuron_weights[i] + self.neighborhood[i] * self.alpha * (
                        inp - self.neuron_weights[i])

    def dead_neurons(self):
        for i in range(len(self.potential)):
            if i == self.winner:
                self.potential[i] = self.potential[i] - self.p_min
            else:
                self.potential[i] = self.potential[i] + (1 / self.number_of_neurons)

    def neuron_activation(self):
        for i in range(len(self.potential)):
            if self.potential[i] < self.p_min:
                self.activation[i] = 0
            else:
                self.activation[i] = 1

    def calculateError(self):
        error = 0
        errorDist = []
        for inp in self.input_data:
            for i in self.neuron_weights:
                errorDist.append(distance.euclidean(i, inp))
            error += min(errorDist) ** 2
            errorDist.clear()
        self.error.append(error / len(self.input_data))

    def train(self, epoch_number):
        self.plot("Before")
        self.all_steps = epoch_number * len(self.input_data)
        combined_data = list(self.input_data)
        step = 0
        self.calculateError()
        for epoch in range(epoch_number):
            np.random.shuffle(combined_data)
            for inp in combined_data:
                self.calculate_distance(inp=inp, for_calculate=self.neuron_weights,
                                        distance_calculation=self.distance)
                self.find_winner()
                self.calculate_distance(inp=self.neuron_weights[self.winner], for_calculate=self.neuron_weights,
                                        distance_calculation=self.winner_distance)
                self.kohonen_neighborhood()
                self.dead_neurons()
                self.update_weights(inp=inp)
                self.clear_lists(step=step)
                self.animation_plots.append(np.copy(self.neuron_weights))
                step += 1
            self.calculateError()
        self.plot("After")
        self.plot_for_error(epoch_number + 1)
        self.animate_plots()


    def plot(self, title):
        input_x = []
        input_y = []
        for i in self.input_data:
            input_x.append(i[0])
            input_y.append(i[1])
        plt.plot(input_x, input_y, 'bo')
        weights_x = []
        weights_y = []
        for i in self.neuron_weights:
            weights_x.append(i[0])
            weights_y.append(i[1])
        plt.plot(weights_x, weights_y, 'bo', color='red')
        plt.title(title)
        plt.show()

    def plot_for_error(self, epoch):
        epoch_range = np.arange(1, epoch + 1, 1)
        plt.plot(epoch_range, self.error, 'ro', markersize=1)
        plt.title("Quantization error")
        plt.show()

    def animate_plots(self):
        fig, ax = plt.subplots()
        ax.axis([np.min(self.animation_plots[0], axis=0)[0] - 3, np.max(self.animation_plots[0], axis=0)[0] + 3,
                 np.min(self.animation_plots[0], axis=0)[1] - 3, np.max(self.animation_plots[0], axis=0)[1] + 3])
        ax.plot(self.input_data[:, 0], self.input_data[:, 1], 'bo')
        line, = ax.plot([], [], 'ro')

        def animate(frame):
            if frame > len(self.animation_plots) - 1:
                frame = len(self.animation_plots) - 1
            line.set_data(self.animation_plots[frame][:, 0], self.animation_plots[frame][:, 1])
            ax.set_title("Input Data: " + str((frame + 1)))
            return line

        ani = animation.FuncAnimation(fig, animate, len(self.animation_plots), interval=1, repeat=False)
        plt.show()


# GeneratePoints.find_points()
SOM = SelfOrganizingMap(number_of_neurons=20, input_data_file="randomPoints.txt", radius=0.5, alpha=0.5, gaussian=0)
SOM.train(20)
