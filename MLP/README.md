## [MLP](https://github.com/JuliaSzymanska/Artificial-Intelligence/tree/master/MLP)
This task consists of two parts, each of which presents one example of how to use the multilayer perceptron (MLP).

### [Approximation](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/MLP/Approximation.py)
Implementation of a neural network with one input layer, one linear hidden layer, and one output layer with a sigmoidal activation function.

#### USAGE
* Create a network instance
```python
network = NeuralNetwork(number_of_input=1, number_of_hidden=10, number_of_output=1, 
                        input_data_file="Approximation_data_1.txt", is_bias=1)
```
* Train network
```python
network.train(epoch_number=2000)
```

#### Output
* Training and testing function approximation
Example plots:
<table>
  <tr>
    <td><img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/MLP/.readme/ApproximationTrainingFunction.png" width="480"></td>
    <td><img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/MLP/.readme/ApproximationTestingFunction.png" width="480"></td>
  </tr>
 </table>
 
* Mean square error plot for training and testing data
Example plot:
<table>
  <tr>
    <td><img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/MLP/.readme/ApproximationErrorForTraining.png" width="480"></td>
    <td><img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/MLP/.readme/ApproximationErrorForTesting.png" width="480"></td>
  </tr>
 </table>
 
* Error for training and testing data:
Example values:
```text
Error for training data:  0.10014346160610355
Error for testing data:  [0.05365087]
```

### [Classification](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/MLP/Classification.py)
Implementation of a neural network with input layer with 1 to 4 neurons, one sigmoidal hidden layer, and one output layer with 3 neurons with a sigmoidal activation function.

#### USAGE
* Create a network instance
```python
network = NeuralNetworkApproximation(number_of_input=4, number_of_hidden=3, number_of_output=4, input_data_file="transformation.txt",
                        expected_data_file="transformation.txt", is_bias=1)
```
* Train network
```python
network.train(epoch_number=2000)
```

#### Output
* Mean square error plot
Example plot:
<p align="center">
  <img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/MLP/.readme/ClassificationPlot.png" width="640">
</p>

* Output data on over-trained network
For input:
```text
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
```
Example results:
```text
[0.98646489 0.00998248 0.00769733 0.00780114]
[0.00794265 0.9872775  0.0056632  0.00718965]
[0.01063865 0.00948483 0.98665415 0.00992539]
[0.00608939 0.0063894  0.00533386 0.98834747]
```
