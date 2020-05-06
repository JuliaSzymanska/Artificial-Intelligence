import csv
import numpy as np
import math

import GeneratePoints


class SelfOrganizingMap(object):
    def __init__(self, x, y, input_data_file):
        self.x = x
        self.y = y
        self.shape = (x, y)
        self.input_data = self.file_input(input_data_file)
        self.neuron_weights = []
        self.initialze_weights()
        self.distance = []

    def initialze_weights(self):
        self.neuron_weights = 2 * np.random.random((self.x, self.y, len(self.input_data[0]))) - 1

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

    def train(self, epoch_number):
        combined_data = list(self.input_data)
        for epoch in range(epoch_number):
            np.random.shuffle(combined_data)
            for inp in combined_data:
                for i in range(len(self.neuron_weights)):
                    self.distance.append([])
                    for j in self.neuron_weights[i]:
                        self.distance[i].append(math.sqrt(math.fabs(j[0] * inp[0]) + (j[1] + inp[1])))
                self.distance = []


# GeneratePoints.findPoints()
SOM = SelfOrganizingMap(3, 4, "RandomPoints.txt")
SOM.train(2)
