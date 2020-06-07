## [MLP](https://github.com/JuliaSzymanska/Artificial-Intelligence/tree/master/MLP)
This task consists of two parts, each of which presents one example of how to use the multilayer perceptron (MLP).
### [Approximation](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/MLP/Approximation.py)
Implementation of a neural network with one input layer, one linear hidden layer, and one output layer with a sigmoidal activation function.

#### USAGE
* Create a network instance
```python
network = NeuralNetwork(number_of_input=4, number_of_hidden=3, number_of_output=4, input_data_file="transformation.txt",
                        expected_data_file="transformation.txt", is_bias=1)
```
* Train network
```python
network.train(epoch_number=2000)
```

#### Output
* mean square error plot
![]()




### [Classification](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/MLP/Classification.py)

