## [K-means](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/K-means/KMeans.py)
The network implementing the classic self-organizing map consists of a set of neurons. Neurons compete with each other for the right to represent the input pattern, which means that the one who responds most strongly is ultimately accepted as a representative of the pattern. The neuron transmission function is a function of the distance between the input vector and the weight vector, and a given neuron can be interpreted as a point / vector in the space of input standards. Finding the optimal distribution of neurons in space is possible in two ways:

#### Usage
* Generate points on the figure
```python
GeneratePoints.find_points()
```
* Create a network instance
```python
Means = KMeans(k=10, input_data_file="Data/RandomPoints.txt", epsilon=0.0001, rand_number=5)
```
* Train network
```python
Means.train()
```

#### Output
* Example plots for data before and after training
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

* Number of epochs to achieve center stabilization
```text
Number of epoch:  22
```

* Value of mean square error for last epoch
Example error:
```text
Final mean square error:  0.08222274965956247
```