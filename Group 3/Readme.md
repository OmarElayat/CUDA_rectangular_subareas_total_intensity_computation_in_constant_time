## Overview
The program performs 2D convolution on greyscale images of arbitrary size. It supports the following eight operations:
-	Blur
-	Emboss
-	Outline
-	Sharpen
-	Sobel (left, right, top, and bottom).

The parallel version of the program uses tiling (8x8 tiles) and constant memory and rely on general caching for halo-cells. 
Boundary conditions are handled by replicating the edge values. 

## How to run
The image file name and the selected operation are provided by the user. 
The program then opens the file converting it to a 2D array, apply convolution to it based on the selected mask, and save the result as a new image 
file. 


## The Masks
Here are the masks to be used for each of the requested operations:

![image](https://user-images.githubusercontent.com/107650627/211076449-62a6e951-9803-4c98-9c30-c5089fa38262.png)

**For a visual (interactive) explanation of the different convolution operations and their effect on image, you can check this URL: http://setosa.io/ev/image-kernels**
