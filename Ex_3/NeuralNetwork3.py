import math

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.spatial import distance


# learning_coeff = 0.45
#
# epoch_error = 0.0
#
# momentum_coeff = 0.9


class NeuralNetwork(object):
    def __init__(self, number_of_radial, number_of_linear, input_data_file, is_bias=0):
        np.random.seed(30)
        self.radial_layer_weights = []
        self.linear_layer_weights = []
        self.number_of_radial = number_of_radial
        self.number_of_linear = number_of_linear
        self.is_bias = is_bias
        self.input_data, self.expected_data = self.file_input(input_data_file)
        self.initialze_weights()
        self.radial_coefficient = []
        self.set_radial_coefficient()
        # self.delta_weights_output_layer = []
        # self.delta_weights_hidden_layer = []
        # self.epoch_error = 0.0
        # self.error_for_epoch = []
        # self.epoch_for_error = []
        # self.expected_data = self.file_input(expected_data_file)

    def initialze_weights(self):
        input = self.input_data
        np.random.shuffle(input)
        for i in range(self.number_of_radial):
            self.radial_layer_weights[i] = input[i]
        self.linear_layer_weights = 2 * np.random.random(
            (self.number_of_radial + self.is_bias, self.number_of_linear)) - 1

    # def calculateDistance(self, inp, forCalculate):
    #     return distance.euclidean(i, inp))

    def set_radial_coefficient(self):
        for i in self.radial_layer_weights:
            max = 0
            for j in self.radial_layer_weights:
                neural_distance = distance.euclidean(i, j)
                if neural_distance > max:
                    max = neural_distance
            self.radial_coefficient.append(max / math.sqrt(2 * self.number_of_radial))

    def linear_func(self, x):
        return x

    def linear_derivative(self, x):
        return 1

    # coefficient to wspoclzynnik z self.coefficient dla danego neuronu
    def rbf_gaussian(self, input, radial_weight, coefficient):
        return np.exp(-1 * ((distance.euclidean(input, radial_weight)) ** 2) / (2 * coefficient ** 2))

    # def backward_propagation(self, radial_layer_result, linear_layer_result, input_data, output_data):
    # avr_err = 0.0
    # output_difference = output_layer_result - output_data  # required for mean squared error
    #
    # for i in output_difference:
    #     avr_err += i ** 2
    # avr_err /= 2
    # self.epoch_error += avr_err
    #
    # delta_coefficient_outp = output_difference * self.sigmoid_derivative(output_layer_result)
    # hidden_layer_error = delta_coefficient_outp.dot(self.output_layer_weights.T)
    # if self.is_bias == 1:
    #     hidden_layer_error = hidden_layer_error[1:]
    #     delta_coefficient_hidden = hidden_layer_error * self.sigmoid_derivative(hidden_layer_result[1:])
    # else:
    #     delta_coefficient_hidden = hidden_layer_error * self.sigmoid_derivative(hidden_layer_result)
    #
    # output_adj = []
    # hidden_adj = []
    # for i in delta_coefficient_outp:
    #     output_adj.append(hidden_layer_result * i)
    # for i in delta_coefficient_hidden:
    #     hidden_adj.append(input_data * i)
    #
    # hidden_adj = np.asarray(hidden_adj)
    # output_adj = np.asarray(output_adj)
    #
    # actual_hidden_adj = (learning_coeff * hidden_adj.T + momentum_coeff * self.delta_weights_hidden_layer)
    # actual_output_adj = (learning_coeff * output_adj.T + momentum_coeff * self.delta_weights_output_layer)
    # self.hidden_layer_weights -= actual_hidden_adj
    # self.output_layer_weights -= actual_output_adj
    #
    # self.delta_weights_hidden_layer = actual_hidden_adj
    # self.delta_weights_output_layer = actual_output_adj

    def train(self, epoch_count):
        combined_data = list(zip(self.input_data, self.expected_data))
        for epoch in range(epoch_count):
            np.random.shuffle(combined_data)
            for inp, outp in combined_data:
                radial_layer_output, linear_layer_output = self.feed_forward(inp)
                self.backward_propagation(radial_layer_output, linear_layer_output, inp, outp)
            self.epoch_error /= 4
            self.epoch_for_error.append(epoch_count)
            self.error_for_epoch.append(self.epoch_error)
            # print(epoch_count, "  ", self.epoch_error)
            epoch_count += 1
        print(epoch_count)

    def file_input(self, file_name):
        with open(file_name, "r") as f:
            expected_val = []
            input_arr = []
            data = csv.reader(f, delimiter=' ')
            for row in data:
                tmp_arr = []
                for i in row:
                    tmp_arr.append(float(i))
                expected_val.append(float(row[1]))
                input_arr.append(tmp_arr)
        return np.asarray(input_arr), np.asarray(expected_val)

    # def graph(self):
    #     plt.plot(self.epoch_for_error, self.error_for_epoch, 'ro', markersize=1)
    #     plt.title("Błąd średniokwadratowy w zależności od epoki")
    #     plt.ylabel("Błąd średniokwadratowy")
    #     plt.xlabel("Epoka")
    #     plt.show()


NeuNet = NeuralNetwork(4, 3, 4, "transformation.txt", "transformation.txt", 1)
NeuNet.train(2000)
# print("Wynik:")
# for i in NeuNet.input_data:
#     print(NeuNet.feed_forward(i, True)[1])
# NeuNet.graph()
