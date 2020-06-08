import csv
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import GeneratePoints


class KMeans(object):
    def __init__(self, k, input_data_file, epsilon, rand_number):
        self.k = k
        self.epsilon = epsilon
        self.input_data = self.file_input(input_data_file)
        self.centroids_weights = self.inizialize_weights(rand_number)
        self.distance = []
        self.winner = []
        np.random.seed(20)
        self.combined_data = list(self.input_data)
        self.flag = True
        self.error = []

    def inizialize_weights(self, rand_number):
        errors = []
        weights = []
        for i in range(rand_number):
            weights.append(np.random.normal(np.mean(self.input_data), np.std(self.input_data),
                                                 size=(self.k, len(self.input_data[0]))))
            errors.append(self.calculate_error(self.input_data, weights[i]))
        return weights[errors.index(min(errors))]

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

    def calculate_distance(self, inp):
        for i in self.centroids_weights:
            self.distance.append(distance.euclidean(i, inp))

    def find_winner(self):
        self.winner.append(self.distance.index(min(self.distance)))

    def update_weights(self):
        avg_x = 0
        avg_y = 0
        counter = 0
        for i in range(len(self.centroids_weights)):
            for j in range(len(self.winner)):
                if i == self.winner[j]:
                    avg_x += self.combined_data[j][0]
                    avg_y += self.combined_data[j][1]
                    counter += 1
            if counter != 0:
                if self.centroids_weights[i][0] - avg_x / counter > self.epsilon \
                        and self.centroids_weights[i][1] - avg_y / counter > self.epsilon:
                    self.flag = True
                self.centroids_weights[i][0] = avg_x / counter
                self.centroids_weights[i][1] = avg_y / counter
            avg_x = 0
            avg_y = 0
            counter = 0

    def calculate_error(self, input, weights):
        error = 0
        error_dist = []
        for inp in input:
            for i in weights:
                error_dist.append(distance.euclidean(i, inp))
            error += min(error_dist) ** 2
            error_dist.clear()
        return error / len(input)

    def train(self):
        counter = 0
        while self.flag:
            self.flag = False
            np.random.shuffle(self.combined_data)
            for inp in self.combined_data:
                self.calculate_distance(inp=inp)
                self.find_winner()
                self.distance.clear()
            if counter == 0:
                self.plot(counter, 1)
            self.update_weights()
            if self.flag:
                self.winner.clear()
            self.error.append(self.calculate_error(input=self.input_data, weights=self.centroids_weights))
            counter += 1
        print("Number of epoch: ", counter)
        print("Final mean square error: ", self.error[-1])
        self.plot(counter, 1, True)
        self.plot_for_error(counter)

    def plot(self, title, color, is_last = False):
        colors = ['#116315', '#FFD600', '#FF6B00', '#5199ff', '#FF2970', '#B40A1B', '#E47CCD', '#782FEF', '#45D09E', '#FEAC92']
        input_x = []
        input_y = []
        if color:
            for j in range(self.k):
                for i in range(len(self.winner)):
                    if self.winner[i] == j:
                        input_x.append(self.combined_data[i][0])
                        input_y.append(self.combined_data[i][1])
                plt.plot(input_x, input_y, color=colors[j], marker='o', ls='')
                input_x.clear()
                input_y.clear()
        else:
            for i in self.input_data:
                input_x.append(i[0])
                input_y.append(i[1])
            plt.plot(input_x, input_y, 'bo')
        weights_x = []
        weights_y = []
        for i in self.centroids_weights:
            weights_x.append(i[0])
            weights_y.append(i[1])
        plt.plot(weights_x, weights_y, 'ko', markersize=9, marker='o')
        for i in range(len(self.centroids_weights)):
            plt.plot(weights_x[i], weights_y[i], color=colors[i], marker='o', ls='', markersize=5)
        if title == 0:
            title = "Before"
        elif is_last:
            title = "After"
        plt.title(title)
        plt.show()

    def plot_for_error(self, epoch):
        epoch_range = np.arange(1, epoch + 1, 1)
        plt.plot(epoch_range, self.error, 'ro', markersize=5)
        plt.title("Quantization error")
        plt.show()