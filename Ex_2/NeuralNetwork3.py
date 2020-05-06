import csv
import numpy as np
import GeneratePoints


class selfOrganizingMap(object):
    def __init__(self, x, y, input_data_file):
        self.x = x
        self.y = y
        self.shape = (x,y)
        self.input_data = self.file_input(input_data_file)
        self.neuron_weights = []
        self.initialze_weights(x, y)


    def initialze_weights(self, x, y):
        self.neuron_weights = 2 * np.random.random((x, y)) - 1

    def file_input(self, file_name):
        input_arr = []
        with open(file_name, "r") as f:
            data = csv.reader(f, delimiter=',')
            for row in data:
                tmp_arr = []
                for i in row:
                    tmp_arr.append(float(i))
                input_arr.append(tmp_arr)
        return np.asarray(input_arr)

# GeneratePoints.findPoints()
selfOrganizingMap(3, 4, "RandomPoints.txt")