## Overview
This program computes the total intensity of one or more rectangular subareas of a greyscale image of arbitrary size. The program uses the summed-area table (aka integral image) to do this computation in
constant time. The image file name and the coordinates of the pixels delimiting the rectangular subareas should be provided by the user. 
The program opens the file converting it to a 2D array, compute the corresponding summed-area table, and finally use the obtained table to quickly compute the total intensity of the
areas specified by the user in O(n) complexity.

## Background Info.
The summed-area table is a 2D array S of the same size as the original image 2D array I such that the value of each S element at a position (x,y) is the sum of the intensities of all pixels in I with lower coordinates (i.e., all the pixels above and to the left of the pixel at (x,y)

![image](https://user-images.githubusercontent.com/107650627/209829928-b32e2786-2f2a-409e-9d0b-6b1afb04982b.png)

So, the summed-area table computation can be seen as a 2D extension of the prefix-sum (scan) computation. As such the program uses a 2D extension of the hierarchal approach using Brent-kung and Kogge-stone algorithms.

So, if the original image ![image](https://user-images.githubusercontent.com/107650627/209830396-a02f6225-b1e1-4468-957c-d4afcc4f75f7.png)

The summed-area table S will be ![image](https://user-images.githubusercontent.com/107650627/209830495-034c3cb9-9015-42e0-86b5-23ee68d0e394.png)

**For more information about summed-area tables**, check this URL: https://computersciencesource.wordpress.com/2010/09/03/computer-vision-the-integral-image/
