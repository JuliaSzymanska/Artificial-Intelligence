import csv
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

import GeneratePoints

np.random.seed(20)


class SelfOrganizingMap(object):
    def __init__(self, k, input_data_file):
        self.k = k
        self.input_data = self.file_input(input_data_file)
        self.centroidsWeights = np.random.normal(np.mean(self.input_data), np.std(self.input_data),
                                                 size=(self.k, len(self.input_data[0])))
        self.distance = []
        self.winner = []
        self.combinedData = list(self.input_data)
        self.flag = True

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

    def calculateDistance(self, inp):
        for i in self.centroidsWeights:
            self.distance.append(distance.euclidean(i, inp))

    def findWinner(self):
        self.winner.append(self.distance.index(min(self.distance)))

    def updateWeights(self):
        avgX = 0
        avgY = 0
        counter = 0
        for i in range(len(self.centroidsWeights)):
            for j in range(len(self.winner)):
                if i == self.winner[j]:
                    avgX += self.combinedData[j][0]
                    avgY += self.combinedData[j][1]
                    counter += 1
            if counter != 0:
                if avgX != 0 and avgY != 0:
                    self.flag = True
                self.centroidsWeights[i][0] = avgX / counter
                self.centroidsWeights[i][1] = avgY / counter
            avgX = 0
            avgY = 0
            counter = 0

    def train(self):
        self.plot("Befor")
        counter = 0
        while self.flag:
            self.flag = False
            np.random.shuffle(self.combinedData)
            for inp in self.combinedData:
                self.calculateDistance(inp)
                self.findWinner()
                self.distance.clear()
            self.updateWeights()
            self.winner.clear()
            self.plot(counter)
            counter += 1

    def plot(self, title):
        inputX = []
        inputY = []
        for i in self.input_data:
            inputX.append(i[0])
            inputY.append(i[1])
        plt.plot(inputX, inputY, 'bo')
        weightsX = []
        weightsY = []
        for i in self.centroidsWeights:
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
# SOM = SelfOrganizingMap(3, "test.txt")
SOM = SelfOrganizingMap(30, "testData.txt")
SOM.train()
