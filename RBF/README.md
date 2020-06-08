## RBF - radial basis function
The RBF network is a neural network with radial base functions. It consists of two layers. 
* First one is a radial layer with radial neurons. Neurons are implemented with a Gaussian radial function, i.e. neurons that calculate the distance between the input vector and the vector representing the center of the radial function, and then this distance multiplied by the radial factor. 

* The second layer consists of traditional linear neurons, i.e. neurons with an identical activation function.

### [Approximation](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/RBF/Approximation.py)

#### Radial neuron's weights

* Draw the weights values from the input data
* Draw the weights values from the input data & change their values in backpropagation

#### Usage

* Create a network instance
```python
network = RBF(number_of_radial=30, number_of_linear=1, input_data_file="Data/approximation_1.txt", is_bias=1,
              is_derivative=1)
```
* Train network
```python
network.train(50)
```

#### Output

* Testing function approximation

Example plot:
<p align="center">
  <img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/RBF/.readme/ApproximationTestFun.png" width="640">
</p>
 
* Mean square error plot for training and testing data

Example plots:
<table>
  <tr>
    <td><p align="center"><img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/RBF/.readme/ApproximationTrainingError.png" width="640"></p></td>
    <td><p align="center"><img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/RBF/.readme/ApproximationTestingError.png" width="640"></p></td>
  </tr>
 </table>
 
* Error for training data:

Example value:
```text
Mean square error for last epoch:  0.0956750012575332
```

### [Classification](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/RBF/Classification.py)

#### Radial neuron's weights

* Draw the weight's values & train them by neural gas
* Draw the weight's values & train them by neural gas & change their values in backpropagation

#### Usage

* Create a network instance
```python
network = Classification(number_of_radial=10, number_of_linear=1, number_of_class=3,
                         input_data_file="Data/classification_train.txt", is_bias=1, is_derivative=1)
```
* Train network
```python
network.train(100)
```

#### Output

* Classification plot

Example plot:
<p align="center">
  <img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/RBF/.readme/Classification.png" width="640">
</p>

* Mean square error plots for training and testing data

Example plots:
<table cellpadding="0" cellspacing="0" border="0">
  <tr>
    <td><p align="center"><img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/RBF/.readme/ClassificationTrainingError.png" width="640"></p></td>
    <td><p align="center"><img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/RBF/.readme/ClassificationTestingError.png" width="640"></p></td>
  </tr>
 </table>

* Confusion matrix for training and testing data

Example matrices:
```text
Confusion matrix for training data:
 [[30.  0.  0.]
 [ 0. 29.  1.]
 [ 0.  1. 29.]]

Confusion matrix for testing data:
 [[31.  0.  0.]
 [ 0. 30.  1.]
 [ 0.  4. 27.]]
```

* Value of mean square error for training and testing data

Example values:
```text
Mean square error for last epoch:  0.019523783180768573
Error for testing data:  [0.02199689]
```
