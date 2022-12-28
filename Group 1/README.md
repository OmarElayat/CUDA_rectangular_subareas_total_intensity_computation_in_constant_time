## Overview
This folder contains 3 programs

### Program 1
Performs vector addition such that each thread is responsible for computing four adjacent elements in the output vector instead of one. The vectors size as well as data are randomly generated.

### Program 2
Randomly generates a grayscale picture by generating a 2D array of integers of size 1000x800 randomly initialized to values ranging between 0 and 255. 
The program uses a CUDA kernel with a 2D grid and 2D blocks to multiply each pixel of the picture by 3 (trimming the resulting value to 255 if it exceeds that
value). Each block have 16x16 threads and each thread is responsible for a single pixel.

### Program 3
Performs matrix addition on square matrices such that each thread is responsible for computing one column of the output (sum) matrix. The size of the matrices and their values are randomly
selected.
