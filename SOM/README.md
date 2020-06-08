## SOM - Self-organizing map
The network implementing the classic self-organizing map consists of a set of neurons. Neurons compete with each other for the right to represent the input pattern, which means that the one who responds most strongly is ultimately accepted as a representative of the pattern. The neuron transmission function is a function of the distance between the input vector and the weight vector, and a given neuron can be interpreted as a point / vector in the space of input standards. Finding the optimal distribution of neurons in space is possible in two ways:

### [Kohonen map](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/SOM/KohonenMap.py)
The classic Kohonen algorithm assumes the adaptation of the weights of only the winning neuron and neurons located no further than the given neighborhood radius. It is also possible to adapt using the Gaussian neighborhood function, determining the learning rate of neurons losing competition in the function of their distance from the winner. In addition, the phenomenon of the appearance of dead neurons was considered, taking into account the activity of neurons in the learning process.

#### Usage
* Generate points on the figure
```python
GeneratePoints.find_points()
```
* Create a network instance
```python
SOM = SelfOrganizingMap(number_of_neurons=20, input_data_file="Data/randomPoints.txt", radius=0.5, alpha=0.5, gaussian=0)
```
* Train network
```python
SOM.train(20)
```

#### Output
* Plots for data before and after training

Example plots:
<table cellpadding="0" cellspacing="0" border="0">
  <tr>
    <td><img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/SOM/.readme/KohonenMapBefore.png" width="640"></td>
    <td><img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/SOM/.readme/KohonenMapAfter.png" width="640"></td>
  </tr>
 </table>

* Quantization error

Example quantization error plot:

<p align="center">
  <img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/SOM/.readme/KohonenMapError.png" width="640">
</p>

### [Neural gas](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/SOM/NeuralGas.py)
The neural gas algorithm assumes the adaptation of the weights of only the winning neuron and the neurons located no further than the specified neighborhood radius. The issue of neighborhood is made by ordering neurons in series depending on the distance of their weight vectors from the given input vector. The learning coefficient is determined in this case based on the position in the series, not the actual distance.

#### Usage
* Generate points on the figure
```python
GeneratePoints.find_points()
```
* Create a network instance
```python
SOM = SelfOrganizingMap(number_of_neurons=20, input_data_file="Data/randomPoints.txt", radius=0.5, alpha=0.5)
```
* Train network
```python
SOM.train(20)
```
#### Output
* Plots for data before and after training

Example plots:
<table cellpadding="0" cellspacing="0" border="0">
  <tr>
    <td><img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/SOM/.readme/NeuralGasBefore.png" width="640"></td>
    <td><img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/SOM/.readme/NeuralGasAfter.png" width="640"></td>
  </tr>
 </table>

* Quantization error plot

Example quantization error plot:
<p align="center">
  <img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/SOM/.readme/NeuralGasError.png" width="640">
</p>
