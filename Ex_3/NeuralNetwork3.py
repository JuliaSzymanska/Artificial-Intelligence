import math

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.spatial import distance

learning_coeff = 0.1

# epoch_error = 0.0

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
        self.radial_coefficient = np.ones(number_of_radial)
        # self.set_radial_coefficient()
        self.epoch_error = 0.0
        self.error_for_epoch = []
        self.epoch_for_error = []

    def initialze_weights(self):
        input = np.copy(self.input_data)
        # np.random.shuffle(input)
        for i in range(self.number_of_radial):
            self.radial_layer_weights.append(input[i])
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

    def backward_propagation(self, radial_layer_output, linear_layer_output, input_data, output_data):
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

    def train(self, epoch_count):
        error_test_data_plot = []
        input_data_plot = []
        output_data_plot = []
        test_data, expected_data = self.file_input("approximation_test.txt")
        combined_data = list(zip(self.input_data, self.expected_data))
        for epoch in range(epoch_count):
            self.epoch_error = 0.0
            # np.random.shuffle(combined_data)
            for inp, outp in combined_data:
                radial_layer_output, linear_layer_output = self.feed_forward(inp)
                if epoch == epoch_count - 1:
                    input_data_plot.append(inp)
                    output_data_plot.append(*linear_layer_output)
                self.backward_propagation(radial_layer_output, linear_layer_output, inp, outp)
            self.epoch_error /= self.input_data.shape[0]
            self.epoch_for_error.append(epoch)
            self.error_for_epoch.append(self.epoch_error)
            # error_test_data_plot.append(self.test_network("approximation_test.txt"))
            # self.plot_uni_graph("Błąd średniokwadratowy dla danych testowych", np.arange(0, epoch_count, 1),
            #                     error_test_data_plot,
            #                     "Epoki",
            #                     "Wartość błędu")
            print(epoch, "  ", self.epoch_error)
        self.plot_uni_graph("Błąd średniokwadratowy", self.epoch_for_error, self.error_for_epoch, "Epoki",
                            "Wartość błędu")
        # self.plot_uni_graph_2_functions("Przebieg funkcji treningowej oraz jej aproksymacji", self.input_data[:, 1],
        #                                 self.expected_data, "X", "Y", input_data_plot, output_data_plot,
        #                                 "Funkcja treningowa")
        self.test_network("approximation_test.txt", True)

    def file_input(self, file_name):
        with open(file_name, "r") as f:
            expected_val = []
            input_arr = []
            data = csv.reader(f, delimiter=' ')
            for row in data:
                # tmp_arr = []
                # for i in row:
                #     tmp_arr.append(float(i))
                expected_val.append(float(row[1]))
                input_arr.append(float(row[0]))
        return np.asarray(input_arr), np.asarray(expected_val)

    def plot_uni_graph(self, title, x_val, y_val, x_label, y_label):
        plt.plot(x_val, y_val, 'ro', markersize=1)
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
        # input_data = test_data[:, 0]
        # test_data_bias = np.insert(test_data, 0, 1, axis=1)
        # input_data = np.stack((test_data_bias[:, 0], test_data_bias[:, 1]), axis=1)
        # expected_data = test_data_bias[:, 2]
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


NeuNet = NeuralNetwork(4, 1, "approximation_1.txt", 0)
NeuNet.train(100)
# print("Blad dla testowego: ", NeuNet.test_network("approximation_test.txt", True))
