## Overview

- This program computes the histogram of the color intensities of a greyscale image. 

- The image file name and the number of histogram bins should be provided by the user. 

- The kernel uses a 2D block of threads to compute the histogram. It also uses interleaving and privatization (without aggregation).
