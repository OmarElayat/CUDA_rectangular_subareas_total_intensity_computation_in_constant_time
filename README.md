## Overview
This repository contains the parallel implementations of multiple algorithms on General purpose GPUs using CUDA C. The main purpose of the parallelization is to increase the 
performance of the sequential algorithms by parallelizing them on a massively parallel architecture.

## The programs components

Each program is composed of:

1) A base (sequential) function that implements the required functionality.
2) A CUDA-kernel and a wrapper function that initializes the necessary device data and calls that kernel to implements the same functionality. Error checking for all CUDA API calls were enforced.
3) A full program that calls both functions on the same input data and verifies that they both produce the same output.

## Performance Evaluation

For each program:

* Both the sequential and the parallel versions were timed and the speedup (or slowdown) obtained from parallelization was computed. Timing the parallel version was done in two different ways:
  * Timing the kernel only.
  * Timing the entire wrapper function (including the memory allocation and data transfer
overheads).
* The program also prints the performance of the sequential and parallel versions in GFLOPS.



## Folders Contents.

### Group 1

1) Vector addition.
2) Generating a random grayscale picture with 2D Blocks.
3) Matrix addition.

### Group 2

1) Naive Matrix Multiplication of any two floating-point matrices.
2) Matrix Multiplication with a more optimized approach (Using shared memory and tiling).
3) Matrix Multiplaction with higher thread granularities, shared memory and tiling for better performances.

### Group 3

1) 2D convolution of 8 masks on arbitrary-sized greyscale images.
2) Fast Intensities evaluation using SAT with prefix sum.
3) Computation of the histogram of the color intensities of a greyscale image.

**For more information about each group/task, check the README.md file of each folder**
