## [SOM](https://github.com/JuliaSzymanska/Artificial-Intelligence/tree/master/SOM)
The network implementing the classic self-organizing map consists of a set of neurons. Neurons compete with each other for the right to represent the input pattern, which means that the one who responds most strongly is ultimately accepted as a representative of the pattern. The neuron transmission function is a function of the distance between the input vector and the weight vector, and a given neuron can be interpreted as a point / vector in the space of input standards. Finding the optimal distribution of neurons in space is possible in two ways:

### [Kohonen map](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/SOM/KohonenMap.py)
The classic Kohonen algorithm assumes the adaptation of the weights of only the winning neuron and neurons located no further than the given neighborhood radius. It is also possible to adapt using the Gaussian neighborhood function, determining the learning rate of neurons losing competition in the function of their distance from the winner. In addition, the phenomenon of the appearance of dead neurons was considered, taking into account the activity of neurons in the learning process.

#### USAGE
* Generate points on the figure
```python
GeneratePoints.find_points()
```
* Create a network instance
```python
SOM = SelfOrganizingMap(number_of_neurons=20, input_data_file="randomPoints.txt", radius=0.5, alpha=0.5, gaussian=0)
```
* Train network
```python
SOM.train(20)
```

#### Output
* Training and testing function approximation
Example plots:
![ApproximationTrainingFunction.png](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/MLP/.readme/ApproximationTrainingFunction.png) 
![ApproximationTestingFunction.png](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/MLP/.readme/ApproximationTestingFunction.png) 

* Mean square error plot for training and testing data
Example plot:
![ApproximationErrorForTraining.png](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/MLP/.readme/ApproximationErrorForTraining.png)
![ApproximationErrorForTesting.png](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/MLP/.readme/ApproximationErrorForTesting.png)

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
![ClassificationPlot.png](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/MLP/.readme/ClassificationPlot.png)

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
