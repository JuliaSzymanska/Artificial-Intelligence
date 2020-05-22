import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import distance
import matplotlib


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
        self.input_data = input_data_file
        self.neuron_weights = np.random.normal(np.mean(self.input_data), np.std(self.input_data),
                                               size=(self.numberOfNeurons, len(self.input_data)))
        self.distance = []
        self.winner = -1
        self.neighborhood = []
        self.winnerDistance = []
        self.error = []
        self.potential = np.ones(self.numberOfNeurons)
        self.activation = np.ones(self.numberOfNeurons)
        self.animation_plots = []

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
        self.allSteps = epoch_number * len(self.input_data)
        combined_data = list(self.input_data)
        step = 0
        self.calculateError()
        for epoch in range(epoch_number):
            np.random.shuffle(combined_data)
            for inp in combined_data:
                self.calculateDistance(inp=inp, forCalculate=self.neuron_weights,
                                       distanceCalculation=self.distance)
                self.findWinner()
                if self.typeOfAlgorithm == 0:
                    self.calculateDistance(inp=self.neuron_weights[self.winner], forCalculate=self.neuron_weights,
                                           distanceCalculation=self.winnerDistance)
                    self.kohonenNeighborhood()
                    self.deadNeurons()
                else:
                    self.sortNeurons()
                    self.gasNeighborhood()
                self.updateWeights(inp=inp)
                self.clearLists(step=step)
                self.animation_plots.append(np.copy(self.neuron_weights))
                step += 1




# # GeneratePoints.findPoints()
# # SOM = SelfOrganizingMap(numberOfNeurons=20, input_data_file="RandomPoints.txt", type=1, radius=0.5, alpha=0.5, gaussian=0)
# SOM = SelfOrganizingMap(numberOfNeurons=15, input_data_file="RandomPoints.txt", type=1, radius=5, alpha=0.5, gaussian=0)
# SOM.train(2)
