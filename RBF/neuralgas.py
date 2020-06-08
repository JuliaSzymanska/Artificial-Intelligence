import numpy as np
import math
from scipy.spatial import distance


class SelfOrganizingMap(object):
    def __init__(self, numberOfNeurons, input_data_file, radius, alpha):
        self.radius = radius
        self.maxRadius = radius
        self.minRadius = 0000.1
        self.allSteps = 0
        self.alpha = alpha
        self.maxAlpha = alpha
        self.minAlpha = 0000.1
        np.random.seed(20)
        self.numberOfNeurons = numberOfNeurons
        self.input_data = input_data_file
        self.neuron_weights = np.random.normal(np.mean(self.input_data), np.std(self.input_data),
                                               size=(self.numberOfNeurons, len(self.input_data[0])))
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
        self.winner = self.distance.index(min(self.distance))

    def clearLists(self, step):
        self.distance.clear()
        self.neighborhood.clear()
        self.winnerDistance.clear()
        self.radius = self.maxRadius * (self.minRadius / self.maxRadius) ** (step / self.allSteps)
        self.alpha = self.maxAlpha * (self.minAlpha / self.maxAlpha) ** (step / self.allSteps)

    def updateWeights(self, inp):
        for i in range(len(self.neuron_weights)):
            self.neuron_weights[i] = self.neuron_weights[i] + self.neighborhood[i] * self.alpha * (
                    inp - self.neuron_weights[i])

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
                self.sortNeurons()
                self.gasNeighborhood()
                self.updateWeights(inp=inp)
                self.clearLists(step=step)
                step += 1
