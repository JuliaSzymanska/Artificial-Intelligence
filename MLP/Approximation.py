import numpy as np
import matplotlib.pyplot as plt
import csv

learning_coeff = 0.1
epoch_error = 0.0

momentum_coeff = 0.1


class NeuralNetworkApproximation(object):
    def __init__(self, number_of_input, number_of_hidden, number_of_output, input_data_file, is_bias=0):
        np.random.seed(2)
        self.hidden_layer_weights = []
        self.output_layer_weights = []
        self.delta_weights_output_layer = []
        self.delta_weights_hidden_layer = []
        self.input_data = []
        self.expected_data = []
        self.initialze_weighs(is_bias, number_of_hidden, number_of_input, number_of_output)
        self.is_bias = is_bias
        self.epoch_error = 0.0
        self.error_for_epoch = []
        self.epoch_for_error = []
        self.data = self.file_input(input_data_file)
        self.resolve_bias()



    def linear_func(self, x):
        return x

    def linear_derivative(self, x):
        return 1

    def sigmoid_func(self, x):
        output = 1 / (1 + np.exp(-x))
        return output

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def feed_forward(self, input_data):
        hidden_layer_output = self.sigmoid_func(np.dot(input_data, self.hidden_layer_weights))
        if self.is_bias == 1:
            hidden_layer_output = np.insert(hidden_layer_output, 0, 1)
        output_layer_output = self.linear_func(np.dot(hidden_layer_output, self.output_layer_weights))
        return hidden_layer_output, output_layer_output

    def backward_propagation(self, hidden_layer_result, output_layer_result, input_data, output_data):
        avr_err = 0.0
        output_difference = output_layer_result - output_data
        for i in output_difference:
            avr_err += i ** 2
        avr_err /= 2
        self.epoch_error += avr_err

        delta_coefficient_outp = output_difference * self.linear_derivative(output_layer_result)
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
        error_test_data_plot = []
        input_data_plot = []
        output_data_plot = []
        combined_data = list(zip(self.input_data, self.expected_data))
        for epoch in range(epoch_number):
            self.epoch_error = 0.0
            np.random.shuffle(combined_data)
            for inp, outp in combined_data:
                if self.is_bias == 0:
                    inp = np.asarray([inp])
                outp = np.asarray([outp])
                hidden_layer_output, output_layer_output = self.feed_forward(inp)
                if epoch == epoch_number - 1:
                    input_data_plot.append(inp[1])
                    output_data_plot.append(*output_layer_output)
                self.backward_propagation(hidden_layer_output, output_layer_output, inp, outp)
            self.epoch_error /= len(self.input_data)
            self.epoch_for_error.append(epoch)
            self.error_for_epoch.append(self.epoch_error)
            error_test_data_plot.append(self.test_network("Approximation_data_test.txt"))
        self.plot_uni_graph("Mean square error for test data", np.arange(0, epoch_number, 1),
                            error_test_data_plot,
                            "Epoch",
                            "Error value")
        self.plot_uni_graph("Mean square error", self.epoch_for_error, self.error_for_epoch, "Epoch",
                            "Error value")
        self.plot_uni_graph_2_functions("Training function and its approximation", self.input_data[:, 1],
                                        self.expected_data, "X", "Y", input_data_plot, output_data_plot,
                                        "Training function")
        print("Error for training data: ", self.error_for_epoch[-1])
        print("Error for testing data: ", network.test_network("Approximation_data_test.txt", True))

    def file_input(self, file_name):
        with open(file_name) as f:
            input_val = []
            expected_val = []
            data = csv.reader(f, delimiter=' ')
            for row in data:
                input_val.append(float(row[0]))
                expected_val.append(float(row[1]))
        return np.stack((input_val, expected_val), axis=1)

    def plot_uni_graph(self, title, x_val, y_val, x_label, y_label):
        plt.plot(x_val, y_val, 'ro', markersize=1)
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

    def initialze_weighs(self, is_bias, number_of_hidden, number_of_input, number_of_output):
        self.hidden_layer_weights = 2 * np.random.random((number_of_input + is_bias, number_of_hidden)) - 1
        self.delta_weights_hidden_layer = np.zeros((number_of_input, number_of_hidden))
        self.output_layer_weights = 2 * np.random.random((number_of_hidden + is_bias, number_of_output)) - 1
        self.delta_weights_output_layer = np.zeros((number_of_hidden, number_of_output))

    def resolve_bias(self):
        if self.is_bias == 1:
            self.data = np.insert(self.data, 0, 1, axis=1)
            self.delta_weights_hidden_layer = np.insert(self.delta_weights_hidden_layer, 0, 0, axis=0)
            self.delta_weights_output_layer = np.insert(self.delta_weights_output_layer, 0, 0, axis=0)
            self.input_data = np.stack((self.data[:, 0], self.data[:, 1]), axis=1)
            self.expected_data = self.data[:, 2]
        else:
            self.input_data = self.data[:, 0]
            self.expected_data = self.data[:, 1]

    def test_network(self, test_file, is_graph=False):
        test_data = self.file_input(test_file)
        test_data_bias = np.insert(test_data, 0, 1, axis=1)
        input_data = np.stack((test_data_bias[:, 0], test_data_bias[:, 1]), axis=1)
        expected_data = test_data_bias[:, 2]
        test_output = []
        err = 0.0
        for test_pair in input_data:
            hidden_layer_output_test, output_layer_output_test = self.feed_forward(test_pair)
            test_output.append(output_layer_output_test)
        for i in range(len(test_output)):
            err += (test_output[i] - expected_data[i]) ** 2
        err /= 2
        if is_graph:
            self.plot_uni_graph_2_functions("Testing function and its approximation", test_data[:, 0],
                                            test_data[:, 1], "X",
                                            "Y", test_data[:, 0], test_output, "Testing function")
        return (err / len(test_output))


network = NeuralNetworkApproximation(number_of_input=1, number_of_hidden=10, number_of_output=1,
                                     input_data_file="Approximation_data_1.txt", is_bias=1)
network.train(epoch_number=2000)

