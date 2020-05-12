import csv
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import GeneratePoints


class SelfOrganizingMap(object):
    def __init__(self, k, input_data_file, epsilon, randNumber):
        self.k = k
        self.epsilon = epsilon
        self.input_data = self.file_input(input_data_file)
        self.centroidsWeights = self.inizializeWeights(randNumber)
        self.distance = []
        self.winner = []
        np.random.seed(20)
        self.combinedData = list(self.input_data)
        self.flag = True
        self.error = []

    def inizializeWeights(self, randNumber):
        errors = []
        weights = []
        for i in range(randNumber):
            weights.append(np.random.normal(np.mean(self.input_data), np.std(self.input_data),
                                                 size=(self.k, len(self.input_data[0]))))
            errors.append(self.calculateError(self.input_data, weights[i]))
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
                if self.centroidsWeights[i][0] - avgX / counter > self.epsilon \
                        and self.centroidsWeights[i][1] - avgY / counter > self.epsilon:
                    self.flag = True
                self.centroidsWeights[i][0] = avgX / counter
                self.centroidsWeights[i][1] = avgY / counter
            avgX = 0
            avgY = 0
            counter = 0

    def calculateError(self, input, weights):
        error = 0
        errorDist = []
        for inp in input:
            for i in weights:
                errorDist.append(distance.euclidean(i, inp))
            error += min(errorDist) ** 2
            errorDist.clear()
        return error / len(input)

    def train(self):
        self.plot("Before")
        counter = 0
        while self.flag:
            self.flag = False
            np.random.shuffle(self.combinedData)
            for inp in self.combinedData:
                self.calculateDistance(inp=inp)
                self.findWinner()
                self.distance.clear()
            self.updateWeights()
            self.winner.clear()
            self.plot(counter)
            self.error.append(self.calculateError(input=self.input_data, weights=self.centroidsWeights))
            counter += 1
        print(counter)
        self.plotForError(counter)

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

    def plotForError(self, epoch):
        epochRange = np.arange(1, epoch + 1, 1)
        plt.plot(epochRange, self.error, 'ro', markersize=1)
        plt.title("Blad kwantyzacji")
        plt.show()


# GeneratePoints.findPoints()
SOM = SelfOrganizingMap(k=30, input_data_file="RandomPoints.txt", epsilon=0.01, randNumber=5)
# SOM = SelfOrganizingMap(k=30, input_data_file="testData.txt", epsilon=0.01, randNumber=5)
SOM.train()


