import numpy as np
import math
from scipy.spatial import distance
from PIL import Image
from numpy import asarray


class ImageCompression(object):
    def __init__(self, number_of_neurons, radius, alpha, gaussian, input_file, output_file):
        image = Image.open(input_file)
        self.output_file = output_file
        self.x, self.y = image.size
        self.input_data = asarray(image).reshape(self.x * self.y, 3).tolist()
        self.radius = radius
        self.max_radius = radius
        self.min_radius = 0000.1
        self.all_steps = 0
        self.alpha = alpha
        self.max_alpha = alpha
        self.min_alpha = 0000.1
        self.p_min = 0.75
        np.random.seed(10)
        self.gaussian = gaussian
        self.number_of_neurons = number_of_neurons
        self.neuron_weights = np.random.normal(0, 255,
                                               size=(number_of_neurons, len(self.input_data[0])))
        self.distance = []
        self.winner = -1
        self.neighborhood = []
        self.winner_distance = []
        self.error = []
        self.potential = np.ones(self.number_of_neurons)
        self.activation = np.ones(self.number_of_neurons)

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


    def clear_lists(self, step):
        self.neuron_activation()
        self.distance.clear()
        self.neighborhood.clear()
        self.winner_distance.clear()
        self.radius = self.max_radius * (self.min_radius / self.max_radius) ** (step / self.all_steps)
        self.alpha = self.max_alpha * (self.min_alpha / self.max_alpha) ** (step / self.all_steps)


    def flatten(self, list):
        new_list = []
        for i in list:
            for j in i:
                new_list.append(j)
        return new_list


    def save_image(self):
        output_array = []
        distances = []
        temp = []
        for inp in self.input_data:
            for i in self.neuron_weights:
                distances.append(distance.euclidean(i, inp))
                temp = [int(x) for x in self.neuron_weights[distances.index(min(distances))]]
            output_array.append(temp)
            distances.clear()
        output_array = self.flatten(output_array)
        image_to_save = Image.frombytes("RGB", (self.x, self.y), bytes(output_array))
        image_to_save.save(self.output_file, "JPEG")


    def calculate_error(self):
        error = 0
        error_dist = []
        for inp in self.input_data:
            for i in self.neuron_weights:
                error_dist.append(distance.euclidean(i, inp))
            error += min(error_dist) ** 2
            error_dist.clear()
        self.error.append(error / len(self.input_data))


    def train(self, epoch_number):
        self.all_steps = epoch_number * len(self.input_data)
        combined_data = list(self.input_data)
        step = 0
        self.calculate_error()
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
                step += 1
            self.calculate_error()
        self.save_image()
        print(self.error)