import math
from collections import Counter

import neuralgas
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.spatial import distance

learning_coeff = 0.1
momentum_coeff = 0.2


class NeuralNetwork(object):
    def __init__(self, number_of_radial, number_of_linear, input_data_file, is_bias=0):
        np.random.seed(0)
        self.radial_layer_weights = []
        self.linear_layer_weights = []
        self.delta_weights_linear_layer = []
        self.number_of_radial = number_of_radial
        self.number_of_linear = number_of_linear
        self.is_bias = is_bias
        self.input_data, self.expected_data = self.file_input(input_data_file)
        self.initialze_weights()
        self.radial_coefficient = []
        self.set_radial_coefficient()
        self.epoch_error = 0.0
        self.error_for_epoch = []
        self.epoch_for_error = []

    def initialze_weights(self):
        input = np.copy(self.input_data)
        SOM = neuralgas.SelfOrganizingMap(numberOfNeurons=self.number_of_radial, input_data_file=input, type=1,
                                          radius=0.5, alpha=0.5, gaussian=0)
        SOM.train(50)
        self.radial_layer_weights = SOM.neuron_weights
        # np.random.shuffle(input)
        # for i in range(self.number_of_radial):
        #     self.radial_layer_weights.append(input[i])
        self.linear_layer_weights = 2 * np.random.random(
            (self.number_of_radial + self.is_bias, self.number_of_linear)) - 1
        self.delta_weights_linear_layer = np.zeros((self.number_of_radial + self.is_bias, self.number_of_linear))

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

    def rbf_gaussian(self, input, radial_weight, coefficient):
        return np.exp(-1 * ((distance.euclidean(input, radial_weight)) ** 2) / (2 * coefficient ** 2))

    def feed_forward(self, input_data):
        radial_layer_output = []
        for i in range(len(self.radial_layer_weights)):
            radial_layer_output.append(
                self.rbf_gaussian(input_data, self.radial_layer_weights[i], self.radial_coefficient[i]))
        if self.is_bias == 1:
            radial_layer_output = np.insert(radial_layer_output, 0, 1)
        output_layer_output = self.linear_func(np.dot(radial_layer_output, self.linear_layer_weights))
        return radial_layer_output, output_layer_output

    def backward_propagation(self, radial_layer_output, linear_layer_output, output_data):
        avr_err = 0.0
        output_difference = linear_layer_output - output_data

        for i in output_difference:
            avr_err += i ** 2
        avr_err /= 2
        self.epoch_error += avr_err

        delta_coefficient_outp = output_difference * self.linear_derivative(linear_layer_output)
        output_adj = []
        val = 0
        for i in delta_coefficient_outp:
            # TODO: poprawic zeby bylo inacej a dialalo tak samo
            val = [i * j for j in radial_layer_output]
            output_adj.append(val)
        output_adj = np.asarray(output_adj)

        output_adj = np.asarray(output_adj)
        actual_output_adj = (learning_coeff * output_adj.T + momentum_coeff * self.delta_weights_linear_layer)
        self.linear_layer_weights -= actual_output_adj
        self.delta_weights_linear_layer = actual_output_adj

    def train(self, epoch_count):  # TODO add calssification algorithm
        error_test_data_plot = []
        input_data_plot = []
        output_data_plot = []
        swapped_values = [[0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        classification_1_error = []
        classification_2_error = []
        classification_3_error = []
        expected_classes_amount = list(Counter(self.expected_data).values())

        combined_data = list(zip(self.input_data, self.expected_data))
        for epoch in range(epoch_count):

            epoch_correct = np.zeros(len(self.input_data[0]))  # 0 for total classes

            # learned_classes_amount = np.zeros(len(expected_classes_amount[0]))

            self.epoch_error = 0.0
            np.random.shuffle(combined_data)
            for inp, outp in combined_data:
                radial_layer_output, linear_layer_output = self.feed_forward(inp)

                swapped_expected = swapped_values[outp]

                if outp == np.argmax(linear_layer_output, axis=0):
                    epoch_correct[0] += 1
                    epoch_correct[outp] += 1

                self.backward_propagation(radial_layer_output, linear_layer_output, swapped_expected)
            self.epoch_error /= self.input_data.shape[0]
            self.epoch_for_error.append(epoch)
            self.error_for_epoch.append(self.epoch_error)
            classification_1_error.append(epoch_correct[1]/expected_classes_amount[0])
            classification_2_error.append(epoch_correct[2]/expected_classes_amount[1])
            classification_3_error.append(epoch_correct[3]/expected_classes_amount[2])

            # error_test_data_plot.append(self.test_network("classification_test.txt", False))
            print(epoch, "  ", self.epoch_error)
            print(epoch, "  ", epoch_correct)
        # self.plot_uni_graph("Błąd średniokwadratowy dla danych testowych", np.arange(0, epoch_count, 1),
        #                     error_test_data_plot,
        #                     "Epoki",
        #                     "Wartość błędu")
        self.plot_uni_graph("Błąd średniokwadratowy", self.epoch_for_error, self.error_for_epoch, "Epoki",
                            "Wartość błędu")
        # self.test_network("classification_test.txt", True)

    def file_input(self, file_name):
        with open(file_name, "r") as f:
            expected_val = []
            input_arr = []
            data = csv.reader(f, delimiter=' ')
            for row in data:
                expected_val.append(int(row[-1]))
                input_arr.append(np.float_(row[:-1]))
        return np.asarray(input_arr), np.asarray(expected_val)

    def plot_uni_graph(self, title, x_val, y_val, x_label, y_label):
        plt.plot(x_val, y_val, 'ro', markersize=3)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def plot_uni_graph_2_functions(self, title, x_val, y_val, x_label, y_label, x_val_1, y_val_1, function):
        plt.plot(x_val, y_val, 'ro', markersize=1, label=function)
        plt.plot(x_val_1, y_val_1, 'bo', markersize=1, label='Aproksymacja funkcji')
        plt.title(title)
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def test_network(self, test_file, is_graph=False):
        test_data, expected_data = self.file_input(test_file)
        test_output = []
        err = 0.0
        for test_pair in test_data:
            hidden_layer_output_test, output_layer_output_test = self.feed_forward(test_pair)
            test_output.append(output_layer_output_test)
        for i in range(len(test_output)):
            err += (test_output[i] - expected_data[i]) ** 2
        err /= 2
        if is_graph:
            self.plot_uni_graph_2_functions("Przebieg funkcji testowej oraz jej aproksymacji", test_data,
                                            expected_data, "X",
                                            "Y", test_data, test_output, "Funkcja testowa")
        return (err / len(test_output))


NeuNet = NeuralNetwork(30, 3, "classification_train.txt", 1)
NeuNet.train(1000)
