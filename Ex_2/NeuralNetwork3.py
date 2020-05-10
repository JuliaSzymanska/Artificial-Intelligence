import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.animation as animation
from scipy.spatial import distance
import matplotlib
matplotlib.use("TkAgg")
import GeneratePoints


class SelfOrganizingMap(object):
    def __init__(self, numberOfNeurons, input_data_file, type, radius, alpha, gaussian):
        self.radius = radius
        self.maxRadius = radius
        self.minRadius = 0000.1
        self.allSteps = 0
        self.alpha = alpha
        self.maxAlpha = alpha
        self.minAlpha = 0000.1
        self.pMin = 0.75
        np.random.seed(20)
        self.gaussian = gaussian
        # 0 for Kohenen, 1 for neural gas
        self.typeOfAlgorithm = type
        self.numberOfNeurons = numberOfNeurons
        self.input_data = self.file_input(input_data_file)
        self.neuron_weights = np.random.normal(np.mean(self.input_data), np.std(self.input_data),
                                               size=(self.numberOfNeurons, len(self.input_data[0])))
        self.distance = []
        self.winner = -1
        self.neighborhood = []
        self.winnerDistance = []
        self.testData = self.file_input("testData.txt")
        self.error = []
        self.potential = np.ones(self.numberOfNeurons)
        self.activation = np.ones(self.numberOfNeurons)
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

    def calculateDistance(self, inp, forCalculate, distanceCalculation):
        for i in forCalculate:
            distanceCalculation.append(distance.euclidean(i, inp))

    def findWinner(self):
        if self.typeOfAlgorithm == 0:
            sortedDistance = np.argsort(np.asarray(self.distance))
            for i in sortedDistance:
                if self.activation[i] != 0:
                    self.winner = i
                    break
        else:
            self.winner = self.distance.index(min(self.distance))

    def kohonenNeighborhood(self):
        if self.gaussian == 1:
            for i in self.winnerDistance:
                self.neighborhood.append(math.exp(-1 * (i ** 2) / (2 * self.radius ** 2)))
        else:
            for i in self.winnerDistance:
                if i <= self.radius:
                    self.neighborhood.append(1)
                else:
                    self.neighborhood.append(0)

    def clearLists(self, step):
        self.neuronActivation()
        self.distance.clear()
        self.neighborhood.clear()
        self.winnerDistance.clear()
        self.radius = self.maxRadius * (self.minRadius / self.maxRadius) ** (step / self.allSteps)
        self.alpha = self.maxAlpha * (self.minAlpha / self.maxAlpha) ** (step / self.allSteps)

    def updateWeights(self, inp):
        for i in range(len(self.neuron_weights)):
            if (self.activation[i] == 1 and self.typeOfAlgorithm == 0) or self.typeOfAlgorithm == 1:
                self.neuron_weights[i] = self.neuron_weights[i] + self.neighborhood[i] * self.alpha * (
                        inp - self.neuron_weights[i])

    def deadNeurons(self):
        for i in range(len(self.potential)):
            if i == self.winner:
                self.potential[i] = self.potential[i] - self.pMin
            else:
                self.potential[i] = self.potential[i] + (1 / self.numberOfNeurons)

    def neuronActivation(self):
        for i in range(len(self.potential)):
            if self.potential[i] < self.pMin:
                self.activation[i] = 0
            else:
                self.activation[i] = 1

    def sortNeurons(self):
        self.distance, self.neuron_weights = (list(t) for t in zip(*sorted(zip(self.distance, self.neuron_weights))))

    def gasNeighborhood(self):
        for i in range(len(self.neuron_weights)):
            self.neighborhood.append(math.exp(-i / self.radius))

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
        self.allSteps = epoch_number * len(self.input_data)
        combined_data = list(self.input_data)
        step = 0
        self.calculateError()
        for epoch in range(epoch_number):
            np.random.shuffle(combined_data)
            for inp in combined_data:
                self.calculateDistance(inp, self.neuron_weights, self.distance)
                self.findWinner()
                if self.typeOfAlgorithm == 0:
                    self.calculateDistance(self.neuron_weights[self.winner], self.neuron_weights,
                                           self.winnerDistance)
                    self.kohonenNeighborhood()
                    self.deadNeurons()
                else:
                    self.sortNeurons()
                    self.gasNeighborhood()
                self.updateWeights(inp)
                self.clearLists(step)
                self.animation_plots.append(np.copy(self.neuron_weights))
                step += 1
            self.calculateError()
        self.plot("After")
        self.plotForError(epoch_number + 1)
        self.animate_plots()

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

    def plotForError(self, epoch):
        epochRange = np.arange(1, epoch + 1, 1)
        plt.plot(epochRange, self.error, 'ro', markersize=1)
        plt.title("Blad kwantyzacji")
        plt.show()

    def animate_plots(self):
        fig, ax = plt.subplots()
        # tutaj kreslamy max/min dla wykresu na podstawie 1 epoki
        ax.axis([np.min(self.animation_plots[0], axis=0)[0] - 3, np.max(self.animation_plots[0], axis=0)[0] + 3,
                 np.min(self.animation_plots[0], axis=0)[1] - 3, np.max(self.animation_plots[0], axis=0)[1] + 3])
        ax.plot(self.input_data[:, 0], self.input_data[:, 1], 'bo')
        line, = ax.plot([], [], 'ro')

        def animate(frame):
            if frame > len(self.animation_plots) - 1:
                frame = len(self.animation_plots) - 1
            line.set_data(self.animation_plots[frame][:, 0], self.animation_plots[frame][:, 1])
            ax.set_title("Input Data: " + str((frame+1)))
            return line

        ani = animation.FuncAnimation(fig, animate, len(self.animation_plots), interval=1, repeat=False)
        plt.show()
        # ani.save("output.gif")


# GeneratePoints.findPoints()
SOM = SelfOrganizingMap(20, "RandomPoints.txt", 0, 0.5, 0.5, 0)
# SOM = SelfOrganizingMap(100, "testData.txt", 0, 0.5, 0.5)
SOM.train(10)
