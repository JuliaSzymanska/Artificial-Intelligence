import math
import neuralgas
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.spatial import distance

learning_coeff = 0.1
momentum_coeff = 0.2


class NeuralNetwork(object):
    def __init__(self, number_of_radial, number_of_linear, number_of_class, input_data_file, is_bias=0,
                 is_derivative=0):
        np.random.seed(0)
        self.radial_layer_weights = []
        self.linear_layer_weights = []
        self.number_of_class = number_of_class
        self.delta_weights_linear_layer = []
        self.delta_weights_radial_layer = []
        self.delta_coefficient_radial_layer = []
        self.number_of_radial = number_of_radial
        self.number_of_linear = number_of_linear
        self.is_bias = is_bias
        self.is_derivative = is_derivative
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

    def rbf_gaussian(self, input, radial_weight, coefficient):
        return np.exp(-1 * ((distance.euclidean(input, radial_weight)) ** 2) / (2 * coefficient ** 2))

    def rbf_gaussian_derivative(self, input):
        output = []
        for i in range(len(input[0])):
            output.append(input[:, i] / np.power(self.radial_coefficient, 2))
        return np.asarray(output)

    def rbf_gaussian_derivative_sigma(self, input):
        output = []
        for i in range(len(input[0])):
            output.append(np.power(input[:, i], 2) / np.power(self.radial_coefficient, 3))
        return np.asarray(output).sum(axis=0)

    def feed_forward(self, input_data):
        radial_layer_output = []
        for i in range(len(self.radial_layer_weights)):
            radial_layer_output.append(
                self.rbf_gaussian(input_data, self.radial_layer_weights[i], self.radial_coefficient[i]))
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

        actual_linear_adj = learning_coeff * linear_adj.T + momentum_coeff * self.delta_weights_linear_layer
        self.linear_layer_weights -= actual_linear_adj
        self.delta_weights_linear_layer = actual_linear_adj

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
        confusion_matrix = np.zeros([self.number_of_class, self.number_of_class])
        combined_data = list(zip(self.input_data, self.expected_data))
        expected_amount_of_obj_in_classes = np.zeros([self.number_of_class])
        assigned_amount_of_obj_in_classes_per_epoch = np.zeros([self.number_of_class])
        assigned_amount_of_obj_in_classes = []
        outputs = list(self.expected_data)
        for i in range(len(expected_amount_of_obj_in_classes)):
            expected_amount_of_obj_in_classes[i] = outputs.count(i + 1)
        for epoch in range(epoch_count):
            epoch_correct = np.zeros(len(self.input_data[0]))  # 0 for total classes
            self.epoch_error = 0.0
            np.random.shuffle(combined_data)
            for inp, outp in combined_data:
                radial_layer_output, linear_layer_output = self.feed_forward(inp)
                for i in range(len(linear_layer_output)):
                    if int(round(linear_layer_output[i])) == outp:
                        assigned_amount_of_obj_in_classes_per_epoch[int(round(linear_layer_output[i] - 1))] += 1
                if epoch == epoch_count - 1:
                    confusion_matrix[int(outp) - 1][int(round(linear_layer_output[0] - 1))] += 1
                self.backward_propagation(radial_layer_output, linear_layer_output, inp, outp)
            assigned_amount_of_obj_in_classes.append(assigned_amount_of_obj_in_classes_per_epoch)
            assigned_amount_of_obj_in_classes_per_epoch = np.zeros([self.number_of_class])
            self.epoch_error /= self.input_data.shape[0]
            self.epoch_for_error.append(epoch)
            self.error_for_epoch.append(self.epoch_error)
            error_test_data_plot.append(self.test_network("classification_test.txt", False))
            print(epoch, "  ", self.epoch_error)
        print("Tablica pomylek:\n", confusion_matrix)
        self.plot_number_of_classifications("Klasyfikacja", expected_amount_of_obj_in_classes,
                                            assigned_amount_of_obj_in_classes, "Epoch", "Number")
        self.plot_uni_graph("Błąd średniokwadratowy dla danych testowych", np.arange(0, epoch_count, 1),
                            error_test_data_plot,
                            "Epoki",
                            "Wartość błędu")
        self.plot_uni_graph("Błąd średniokwadratowy", self.epoch_for_error, self.error_for_epoch, "Epoki",
                            "Wartość błędu")
        print("Blad dla danych testowych: ", self.test_network("classification_test.txt", True))

    def file_input(self, file_name):
        with open(file_name, "r") as f:
            expected_val = []
            input_arr = []
            data = csv.reader(f, delimiter=' ')
            for row in data:
                expected_val.append(int(row[-1]))
                input_arr.append(np.float_(row[:-1]))
        return np.asarray(input_arr), np.asarray(expected_val)

    def plot_number_of_classifications(self, title, expected_matrix, actual_matrix, x_label, y_label):
        colors = ['#116315', '#FFD600', '#FF6B00', '#5199ff', '#FF2970', '#B40A1B', '#E47CCD', '#782FEF', '#45D09E',
                  '#FEAC92']
        epoch = []
        for j in range(self.number_of_class):
            inputX = []
            for i in range(len(actual_matrix)):
                inputX.append(actual_matrix[i][j])
                if j == 0:
                    epoch.append(i)
            inputX = np.asarray(inputX)
            inputX = inputX / expected_matrix[j]
            plt.plot(inputX, colors[j], markersize=3, marker='o', ls='', label=str(j + 1))
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend()
        plt.show()

    def plot_uni_graph(self, title, x_val, y_val, x_label, y_label):
        plt.plot(x_val, y_val, 'r', markersize=3)
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

    def test_network(self, test_file, is_last=False):
        test_data, expected_data = self.file_input(test_file)
        confusion_matrix = np.zeros([self.number_of_class, self.number_of_class])
        test_output = []
        err = 0.0
        counter = 0
        for test_pair in test_data:
            hidden_layer_output_test, output_layer_output_test = self.feed_forward(test_pair)
            test_output.append(output_layer_output_test)
            if is_last:
                confusion_matrix[int(expected_data[counter]) - 1][int(round(output_layer_output_test[0] - 1))] += 1
            counter += 1
        for i in range(len(test_output)):
            err += (test_output[i] - expected_data[i]) ** 2
        err /= 2
        if is_last:
            print("Tablica pomylek dla daynch testowych:\n", confusion_matrix)
        return err / len(test_output)


NeuNet = NeuralNetwork(number_of_radial=10, number_of_linear=1, number_of_class=3,
                       input_data_file="classification_train.txt", is_bias=1, is_derivative=1)
NeuNet.train(100)
