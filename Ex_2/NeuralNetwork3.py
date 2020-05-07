import csv
import numpy as np
import math
import matplotlib.pyplot as plt

import GeneratePoints


class SelfOrganizingMap(object):
    def __init__(self, x, y, input_data_file):
        self.lamda = 0.3
        self.alpha = 0.5
        self.x = x
        self.y = y
        self.numberOfNeurons = x * y
        self.input_data = self.file_input(input_data_file)
        self.neuron_weights = np.random.normal(np.mean(self.input_data), np.std(self.input_data),
                                               size=(self.numberOfNeurons, len(self.input_data[0])))
        self.distance = []
        self.minDistance = -1
        self.neighborhood = []
        self.winnerDistance = []
        self.testData = self.file_input("testData.txt")
        self.error = []

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

    def distanceFun(self, x, y):
        return

    def calculateDistance(self, inp, forCalculate, distanseResult):
        for i in forCalculate:
            distanseResult.append(math.sqrt(math.fabs((i[0] - inp[0]) ** 2
                                                      + (i[1] - inp[1]) ** 2)))

    def findMinDistance(self):
        self.minDistance = self.distance.index(min(self.distance))

    def gaussianNeighborhood(self):
        for i in self.winnerDistance:
            self.neighborhood.append(math.exp(-1 * (i ** 2) / (2 * self.lamda ** 2)))

    def clearLists(self):
        self.distance.clear()
        self.neighborhood.clear()
        self.winnerDistance.clear()

    def updateWeights(self, inp):
        for i in range(len(self.neuron_weights)):
            self.neuron_weights[i] = self.neuron_weights[i] + self.neighborhood[i] * self.alpha * (
                    inp - self.neuron_weights[i])

    def train(self, epoch_number):
        self.plot("Befor")
        combined_data = list(self.input_data)
        for epoch in range(epoch_number):
            print(epoch)
            np.random.shuffle(combined_data)
            for inp in combined_data:
                self.calculateDistance(inp, self.neuron_weights, self.distance)
                self.findMinDistance()
                self.calculateDistance(self.neuron_weights[self.minDistance], self.neuron_weights,
                                       self.winnerDistance)
                self.gaussianNeighborhood()
                self.updateWeights(inp)
                self.error.append(self.distance[self.minDistance] / len(self.input_data[0]))
                self.clearLists()
        self.plot("After")

    def plot(self, title):
        inputX = []
        inputY = []
        for i in self.input_data:
            inputX.append(i[0])
            inputY.append(i[1])
        plt.plot(inputX, inputY, 'bo')
        weightsX = []
        weightsY = []
        for i in self.neuron_weights:
            weightsX.append(i[0])
            weightsY.append(i[1])
        plt.plot(weightsX, weightsY, 'bo', color='red')
        plt.title(title)
        plt.show()

    # def plotForError(self):
    #     plt.plot(self.error, self.number, 'ro', markersize=1)
    #     plt.title("Blad kwantyzacji")
    #     plt.xlabel(x_label)
    #     plt.ylabel(y_label)
    #     plt.show()


# GeneratePoints.findPoints()
# SOM = SelfOrganizingMap(3, 4, "RandomPoints.txt")
SOM = SelfOrganizingMap(10, 10, "testData.txt")
SOM.train(1)
