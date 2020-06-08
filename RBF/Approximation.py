import math
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.spatial import distance

learning_coeff = 0.02
momentum_coeff = 0.2


class RBF(object):
    def __init__(self, number_of_radial, number_of_linear, input_data_file, is_bias=0, is_derivative=0):
        np.random.seed(0)
        self.is_derivative = is_derivative
        self.radial_layer_weights = []
        self.linear_layer_weights = []
        self.delta_weights_linear_layer = []
        self.delta_weights_radial_layer = []
        self.delta_coefficient_radial_layer = []
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
        np.random.shuffle(input)
        for i in range(self.number_of_radial):
            self.radial_layer_weights.append(input[i])
        self.linear_layer_weights = 2 * np.random.random(
            (self.number_of_radial + self.is_bias, self.number_of_linear)) - 1
        self.delta_weights_linear_layer = np.zeros((self.number_of_radial + self.is_bias, self.number_of_linear))
        self.delta_weights_radial_layer = np.zeros_like(self.radial_layer_weights)

    def set_radial_coefficient(self):
        for i in self.radial_layer_weights:
            max = 0
            for j in self.radial_layer_weights:
                neural_distance = distance.euclidean(i, j)
                if neural_distance > max:
                    max = neural_distance
            self.radial_coefficient.append(max / math.sqrt(2 * self.number_of_radial))
        if not self.delta_coefficient_radial_layer:
            self.delta_coefficient_radial_layer = np.zeros_like(self.radial_coefficient)

    def linear_func(self, x):
        return x

    def linear_derivative(self, x):
        return 1

    def rbf_gaussian(self, one_input):
        output = []
        for i in range(len(self.radial_layer_weights)):
            output.append(np.exp(-1 * ((distance.euclidean(one_input, self.radial_layer_weights[i])) ** 2) / (
                    2 * self.radial_coefficient[i] ** 2)))
        return output

    def rbf_gaussian_derivative(self, one_input):
        output = [one_input * self.rbf_gaussian(one_input) / np.power(self.radial_coefficient, 2)]
        return np.asarray(output)

    def rbf_gaussian_derivative_sigma(self, one_input):
        output = [np.power(one_input, 2) * self.rbf_gaussian(one_input) / np.power(self.radial_coefficient, 3)]
        return np.asarray(output).sum(axis=0)

    def feed_forward(self, input_data):
        radial_layer_output = self.rbf_gaussian(input_data)
        if self.is_bias == 1:
            radial_layer_output = np.insert(radial_layer_output, 0, 1)
        output_layer_output = self.linear_func(np.dot(radial_layer_output, self.linear_layer_weights))
        return radial_layer_output, output_layer_output

    def backward_propagation(self, radial_layer_output, linear_layer_output, inp, output_data):
        avr_err = 0.0
        output_difference = linear_layer_output - output_data

        for i in output_difference:
            avr_err += i ** 2
        avr_err /= 2
        self.epoch_error += avr_err

        delta_coefficient_linear = output_difference * self.linear_derivative(linear_layer_output)

        linear_adj = np.array([(radial_layer_output * delta_coefficient_linear)])

        actual_output_adj = learning_coeff * linear_adj.T + momentum_coeff * self.delta_weights_linear_layer

        self.linear_layer_weights -= actual_output_adj
        self.delta_weights_linear_layer = actual_output_adj

        if self.is_derivative:
            radial_layer_error = delta_coefficient_linear.dot(self.linear_layer_weights.T)
            if self.is_bias:
                radial_layer_error = radial_layer_error[1:]
                radial_output = radial_layer_output[1:]
            else:
                radial_output = radial_layer_output

            delta_coefficient_radial = radial_layer_error * self.rbf_gaussian_derivative(
                inp - self.radial_layer_weights)

            radial_adj = (radial_output * delta_coefficient_radial).T
            radial_adj = radial_adj.ravel()

            delta_coefficient_sigma = radial_layer_error * self.rbf_gaussian_derivative_sigma(
                inp - self.radial_layer_weights)

            sigma_adj = radial_output * delta_coefficient_sigma

            actual_radial_adj = learning_coeff * radial_adj + momentum_coeff * self.delta_weights_radial_layer
            actual_radial_coefficient_adj = learning_coeff * sigma_adj \
                                            + momentum_coeff * self.delta_coefficient_radial_layer

            self.radial_layer_weights -= actual_radial_adj
            self.radial_coefficient -= actual_radial_coefficient_adj

            self.delta_coefficient_radial_layer = actual_radial_coefficient_adj
            self.delta_weights_radial_layer = actual_radial_adj

    def train(self, epoch_count):
        error_test_data_plot = []
        input_data_plot = []
        output_data_plot = []
        combined_data = list(zip(self.input_data, self.expected_data))
        for epoch in range(epoch_count):
            self.epoch_error = 0.0
            np.random.shuffle(combined_data)
            for inp, outp in combined_data:
                radial_layer_output, linear_layer_output = self.feed_forward(inp)
                if epoch == epoch_count - 1:
                    input_data_plot.append(inp)
                    output_data_plot.append(*linear_layer_output)
                self.backward_propagation(radial_layer_output, linear_layer_output, inp, outp)
            self.epoch_error /= self.input_data.shape[0]
            self.epoch_for_error.append(epoch)
            self.error_for_epoch.append(self.epoch_error)
            error_test_data_plot.append(self.test_network("Data/approximation_test.txt", False))
        print("Mean square error for last epoch: ", self.epoch_error)
        self.plot_uni_graph("Mean square error for testing data", np.arange(0, epoch_count, 1),
                            error_test_data_plot,
                            "Epoch",
                            "Error value")
        self.plot_uni_graph("Mean square error for training data", self.epoch_for_error, self.error_for_epoch, "Epoch",
                            "Error value")
        self.test_network("Data/approximation_test.txt", True)

    def file_input(self, file_name):
        with open(file_name, "r") as f:
            expected_val = []
            input_arr = []
            data = csv.reader(f, delimiter=' ')
            for row in data:
                expected_val.append(float(row[1]))
                input_arr.append(float(row[0]))
        return np.asarray(input_arr), np.asarray(expected_val)

    def plot_uni_graph(self, title, x_val, y_val, x_label, y_label):
        plt.plot(x_val, y_val, 'ro', markersize=3)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def plot_uni_graph_2_functions(self, title, x_val, y_val, x_label, y_label, x_val_1, y_val_1, function):
        plt.plot(x_val, y_val, 'ro', markersize=1, label=function)
        plt.plot(x_val_1, y_val_1, 'bo', markersize=1, label='Approximation of functions')
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
            self.plot_uni_graph_2_functions("Testing function and its approximation", test_data,
                                            expected_data, "X",
                                            "Y", test_data, test_output, "Testing function")
        return (err / len(test_output))
