import numpy as np
import matplotlib.pyplot as plt
import csv

learning_coeff = 0.45

epoch_error = 0.0

momentum_coeff = 0.9


class NeuralNetwork(object):
    def __init__(self, number_of_input, number_of_hidden, number_of_output, input_data_file, expected_data_file,
                 is_bias=0):
        np.random.seed(30)
        self.hidden_layer_weights = []
        self.output_layer_weights = []
        self.delta_weights_output_layer = []
        self.delta_weights_hidden_layer = []
        self.initialze_weights(is_bias, number_of_hidden, number_of_input, number_of_output)
        self.is_bias = is_bias
        self.epoch_error = 0.0
        self.error_for_epoch = []
        self.epoch_for_error = []
        self.input_data = self.file_input(input_data_file)
        self.expected_data = self.file_input(expected_data_file)
        self.resolve_bias()

    def sigmoid_function(self, x):
        result = 1 / (1 + np.exp(-x))
        return result

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feed_forward(self, input_data, is_training=True):
        if not is_training and self.is_bias:
            input_data = np.insert(input_data, 0, 1)
        hidden_layer_output = self.sigmoid_function(np.dot(input_data, self.hidden_layer_weights))
        if self.is_bias == 1:
            hidden_layer_output = np.insert(hidden_layer_output, 0, 1)
        output_layer_output = self.sigmoid_function(np.dot(hidden_layer_output, self.output_layer_weights))
        return hidden_layer_output, output_layer_output

    def backward_propagation(self, hidden_layer_result, output_layer_result, input_data, output_data):
        avr_err = 0.0
        output_difference = output_layer_result - output_data

        for i in output_difference:
            avr_err += i ** 2
        avr_err /= 2
        self.epoch_error += avr_err

        delta_coefficient_outp = output_difference * self.sigmoid_derivative(output_layer_result)
        hidden_layer_error = delta_coefficient_outp.dot(self.output_layer_weights.T)
        if self.is_bias == 1:
            hidden_layer_error = hidden_layer_error[1:]
            delta_coefficient_hidden = hidden_layer_error * self.sigmoid_derivative(hidden_layer_result[1:])
        else:
            delta_coefficient_hidden = hidden_layer_error * self.sigmoid_derivative(hidden_layer_result)

        output_adj = []
        hidden_adj = []
        for i in delta_coefficient_outp:
            output_adj.append(hidden_layer_result * i)
        for i in delta_coefficient_hidden:
            hidden_adj.append(input_data * i)

        hidden_adj = np.asarray(hidden_adj)
        output_adj = np.asarray(output_adj)

        actual_hidden_adj = (learning_coeff * hidden_adj.T + momentum_coeff * self.delta_weights_hidden_layer)
        actual_output_adj = (learning_coeff * output_adj.T + momentum_coeff * self.delta_weights_output_layer)
        self.hidden_layer_weights -= actual_hidden_adj
        self.output_layer_weights -= actual_output_adj


        self.delta_weights_hidden_layer = actual_hidden_adj
        self.delta_weights_output_layer = actual_output_adj

    def train(self, epoch_number):
        combined_data = list(zip(self.input_data, self.expected_data))
        for epoch in range(epoch_number):
            self.epoch_error = 0.0
            np.random.shuffle(combined_data)
            for inp, outp in combined_data:
                hidden_layer_output, output_layer_output = self.feed_forward(inp)
                self.backward_propagation(hidden_layer_output, output_layer_output, inp, outp)
            self.epoch_error /= 4
            self.epoch_for_error.append(epoch)
            self.error_for_epoch.append(self.epoch_error)
        for i in self.input_data:
            print(self.feed_forward(i, True)[1])
        self.graph()

    def initialze_weights(self, is_bias, number_of_hidden, number_of_input, number_of_output):
        self.hidden_layer_weights = 2 * np.random.random((number_of_input + is_bias, number_of_hidden)) - 1
        self.delta_weights_hidden_layer = np.zeros((number_of_input, number_of_hidden))
        self.output_layer_weights = 2 * np.random.random((number_of_hidden + is_bias, number_of_output)) - 1
        self.delta_weights_output_layer = np.zeros((number_of_hidden, number_of_output))

    def resolve_bias(self):
        if self.is_bias == 1:
            self.input_data = np.insert(self.input_data, 0, 1, axis=1)
            self.delta_weights_hidden_layer = np.insert(self.delta_weights_hidden_layer, 0, 0, axis=0)
            self.delta_weights_output_layer = np.insert(self.delta_weights_output_layer, 0, 0, axis=0)

    def file_input(self, file_name):
        input_arr = []
        with open(file_name, "r") as f:
            data = csv.reader(f, delimiter=' ')
            for row in data:
                tmp_arr = []
                for i in row:
                    tmp_arr.append(float(i))
                input_arr.append(tmp_arr)
        return np.asarray(input_arr)

    def graph(self):
        plt.plot(self.epoch_for_error, self.error_for_epoch, 'ro', markersize=1)
        plt.title("Mean square error for the epoch")
        plt.ylabel("Square error")
        plt.xlabel("Epoch")
        plt.show()


network = NeuralNetwork(number_of_input=4, number_of_hidden=3, number_of_output=4, input_data_file="transformation_data.txt",
                        expected_data_file="transformation_data.txt", is_bias=1)
network.train(epoch_number=2000)
