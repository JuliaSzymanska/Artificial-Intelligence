## [Image Compression](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/Image_compression/ImageCompression.py)
Image quantization is a process that reduces the number of image colors, with the intention of making the new image visually similar to the original image. Color quantization is critical to displaying multi-color images on devices that can only display a limited number of colors, usually due to memory limitations, and allow efficient compression of some types of images.

#### Algorithm
* [Kohonen Algorithm](https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/SOM/KohonenMap.py)

#### Usage

* Create a network instance
```python
compression = ImageCompression(number_of_neurons=16, radius=0.5, alpha=0.5,
                       gaussian=0, input_file="Data/Colorful.jpg", output_file=".readme/CompressedColorful.jpeg")
```

* Train network
```python
compression.train(1)
```

#### Output
* Compressed image

Example compression:

<table  cellpadding="0" cellspacing="0" border="0">
  <tr>
    <td>Original colorful image</td>
    <td>Compressed colorful image</td>
  </tr>
  <tr>
    <td><img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/Image_compression/Data/Colorful.jpg" width=270 height=480></td>
    <td><img src="https://github.com/JuliaSzymanska/Artificial-Intelligence/blob/master/Image_compression/.readme/CompressedColorful.jpeg" width=270 height=480></td>
  </tr>
 </table>