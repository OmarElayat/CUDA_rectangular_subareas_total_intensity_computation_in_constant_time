## Overview
This folder contains 3 programs

### Program 1
* Performs matrix multiplication of any two floating-point matrices A and B. A and B do not need
to be square matrices; however, their dimensions should be compatible. i.e, the number of columns of A has to be
equal to the number of rows of B. 
* A and B are initialized to random values by the host, but their dimensions are specified in the program using #define directives. 
* The parallel implementation uses multiple 2D blocks of threads but no shared memory. 
* The program prints the performance of the sequential and parallel versions in GFLOPS. It also Provide results for each of the following block sizes: 16x16, and 32x32 

### Program 2
Repeats Program 1 using shared memory (and tiling) for the parallel implementation.

### Program 3
Repeats Progran 2 with thread granularities of 2 and 4. i.e, each thread block will compute 2 or 4 elements in the output matrix (Each block of threads will be responsible for computing 2 or 4 output tiles).
