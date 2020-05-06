import csv
import numpy as np
import math
import GeneratePoints


class SelfOrganizingMap(object):
    def __init__(self, x, y, input_data_file):
        self.lamda = 1
        self.alpha = 0.5
        self.x = x
        self.y = y
        self.shape = (x, y)
        self.input_data = self.file_input(input_data_file)
        self.neuron_weights = []
        self.initialze_weights()
        self.distance = []
        self.minDistanceX = -1
        self.minDistanceY = -1
        self.neighborhood = []
        self.winnerDistance = []

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

    def distanceFun(self, x, y):
        return math.sqrt(math.fabs((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2))

    def calculateDistance(self, inp, forCalculate, distanseResult):
        for i in range(len(forCalculate)):
            distanseResult.append([])
            for j in forCalculate[i]:
                distanseResult[i].append(self.distanceFun(j, inp))
        return distanseResult

    def findMinDistance(self):
        minV = self.distance[0][0]
        self.minDistanceX = 0
        self.minDistanceY = 0
        for i in self.distance:
            if minV > min(i):
                minV = min(i)
                self.minDistanceX = self.distance.index(i)
                self.minDistanceY = i.index(minV)


    def gaussianFun(self, i, j):
        return math.exp(-1 * (self.winnerDistance[i][j] ** 2) / (2 * self.lamda ** 2))

    def gaussianNeighborhood(self):
        for i in range(len(self.winnerDistance)):
            self.neighborhood.append([])
            for j in range (len(self.winnerDistance[i])):
                self.neighborhood[i].append(self.gaussianFun(i, j))

    def clearLists(self):
        self.distance.clear()
        self.neighborhood.clear()

    def updateWeights(self, inp):
        for i in range(len(self.neuron_weights)):
            for j in range(len(self.neuron_weights[i])):
                self.neuron_weights[i][j] = self.neuron_weights[i][j] + self.neighborhood[i][j] * self.alpha * (inp - self.neuron_weights[i][j])


    def train(self, epoch_number):
        combined_data = list(self.input_data)
        for epoch in range(epoch_number):
            np.random.shuffle(combined_data)
            for inp in combined_data:
                self.calculateDistance(inp, self.neuron_weights, self.distance)
                self.findMinDistance()
                self.calculateDistance(self.neuron_weights[self.minDistanceX][self.minDistanceY], self.neuron_weights, self.winnerDistance)
                self.gaussianNeighborhood()
                self.updateWeights(inp)
                self.clearLists()



# GeneratePoints.findPoints()
SOM = SelfOrganizingMap(3, 4, "RandomPoints.txt")
SOM.train(7)
