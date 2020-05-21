import numpy as np
import matplotlib.pyplot as plt
import csv

# learning_coeff = 0.45
#
# epoch_error = 0.0
#
# momentum_coeff = 0.9


class NeuralNetwork(object):
    def __init__(self, number_of_input, number_of_hidden, number_of_output, input_data_file, expected_data_file,
                 is_bias=0):
        np.random.seed(30)
        # self.hidden_layer_weights = []
        # self.output_layer_weights = []
        # self.delta_weights_output_layer = []
        # self.delta_weights_hidden_layer = []
        # self.initialze_weights(is_bias, number_of_hidden, number_of_input, number_of_output)
        # self.is_bias = is_bias
        # self.epoch_error = 0.0
        # self.error_for_epoch = []
        # self.epoch_for_error = []
        # self.input_data = self.file_input(input_data_file)
        # self.expected_data = self.file_input(expected_data_file)
        # self.resolve_bias()

    def linear_func(self, x):
        return x

    def linear_derivative(self, x):
        return 1

    def train(self, epoch_count):
        combined_data = list(zip(self.input_data, self.expected_data))
        # for epoch in range(epoch_count):
        epoch_count = 0
        self.epoch_error = 1.0
        while self.epoch_error > 0.0001:
            self.epoch_error = 0.0
            np.random.shuffle(combined_data)
            for inp, outp in combined_data:
                hidden_layer_output, output_layer_output = self.feed_forward(inp)
                self.backward_propagation(hidden_layer_output, output_layer_output, inp, outp)
            self.epoch_error /= 4
            self.epoch_for_error.append(epoch_count)
            self.error_for_epoch.append(self.epoch_error)
            # print(epoch_count, "  ", self.epoch_error)
            epoch_count += 1
        print(epoch_count)

    def initialze_weights(self, is_bias, number_of_hidden, number_of_input, number_of_output):
        self.hidden_layer_weights = 2 * np.random.random((number_of_input + is_bias, number_of_hidden)) - 1
        self.delta_weights_hidden_layer = np.zeros((number_of_input, number_of_hidden))
        self.output_layer_weights = 2 * np.random.random((number_of_hidden + is_bias, number_of_output)) - 1
        self.delta_weights_output_layer = np.zeros((number_of_hidden, number_of_output))

    # def resolve_bias(self):
    #     if self.is_bias == 1:
    #         self.input_data = np.insert(self.input_data, 0, 1, axis=1)
    #         self.delta_weights_hidden_layer = np.insert(self.delta_weights_hidden_layer, 0, 0, axis=0)
    #         self.delta_weights_output_layer = np.insert(self.delta_weights_output_layer, 0, 0, axis=0)

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
        plt.title("Błąd średniokwadratowy w zależności od epoki")
        plt.ylabel("Błąd średniokwadratowy")
        plt.xlabel("Epoka")
        plt.show()


NeuNet = NeuralNetwork(4, 3, 4, "transformation.txt", "transformation.txt", 1)
NeuNet.train(2000)
print("Wynik:")
for i in NeuNet.input_data:
    print(NeuNet.feed_forward(i, True)[1])
NeuNet.graph()
