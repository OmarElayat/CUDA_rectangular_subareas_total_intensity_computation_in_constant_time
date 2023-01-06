#include <stdio.h>
#include <iostream>
#include <time.h>
#include <cstdlib>
#include <string>

#define cimg_display 0
#define cimg_use_jpeg
#include "CImg.h"
using namespace cimg_library; 
using namespace std;

#define TILE_WIDTH 8
#define TILE_WIDTH_f float(TILE_WIDTH)


__global__ void histogram_2D_kernel(float* N, unsigned int * histo,int width , int height, int num_bins)
{

    unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    extern __shared__ unsigned int histo_s[];

     if (threadIdx.x < num_bins && threadIdx.y == 0) histo_s[threadIdx.x] = 0;
    __syncthreads(); 
    

    for (unsigned int i=tidy; i<height; i+=blockDim.y * gridDim.y)
    {
        for(unsigned int j=tidx; j<width; j+=blockDim.x * gridDim.x) {
            int curr_pixel = N[i * width + j];
            if(curr_pixel>=0 && curr_pixel <= 255)
                // int ii = curr_pixel/(255/num_bins);
                atomicAdd(&(histo_s[curr_pixel/(255/num_bins)]), 1);
            }
    }


    __syncthreads();
    if (threadIdx.x < num_bins && threadIdx.y == 0)
        atomicAdd(&(histo[threadIdx.x]),histo_s[threadIdx.x]);
}


void histo_2D_OnDevice (float* N, unsigned int * histo, int width , int height, int num_bins, double & gflops)  {


    float *d_N;
    unsigned int *d_histo;
    int size = width * height * sizeof(float);
    int size_histo = num_bins * sizeof(unsigned int);


    cudaError_t err1 = cudaMalloc((void**) &d_N, size);

    if (err1!= cudaSuccess) {
        printf("%s in %s at line %d\n",
        cudaGetErrorString(err1), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy( d_N, N,  size, cudaMemcpyHostToDevice);

    cudaError_t err2 = cudaMalloc((void**) &d_histo, size_histo);
    if (err2!= cudaSuccess) {
        printf("%s in %s at line %d\n",
        cudaGetErrorString(err2), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

   // dim3 dimGrid (ceil(width/8.0), ceil(height/8.0),1);
    dim3 dimGrid (1, 1,1);
    dim3 dimBlock (8,8,1);

    double op_num = (double)((double)height * (double)width);
    clock_t start_dev = clock();
    histogram_2D_kernel<<<dimGrid, dimBlock,size_histo>>>(d_N,  d_histo, width, height, num_bins);
    cudaDeviceSynchronize();
    clock_t stop_dev = clock();
    double time_spent_device = (double)(stop_dev - start_dev) / CLOCKS_PER_SEC;
    gflops = op_num / time_spent_device;

    cudaMemcpy(histo, d_histo, size_histo, cudaMemcpyDeviceToHost);

    cudaFree(d_N); 
    cudaFree(d_histo);
}



void histo_2D_sequential(float* N, unsigned int * histo, int width , int height, int num_bins) {

for (int i = 0; i < height; ++i)              // rows
{
    for (int j = 0; j < width; ++j)          // columns
    {
        float curr = N[i *width + j];
        if (curr >= 0 && curr <= 255)
        {
            int index = curr/(255/num_bins);
            histo[index]++;
        }
    }
}
}


int main (){

    int histo_bin;
    char  file[] = "Cat0.jpeg";
    cout << "enter image file : " ;
    cin>> file;
    //char* filename = "Cat0.jpeg";
    CImg<float> img1(file);
   // CImg<float> img1("Cat0.jpeg");

    cout << "number of histogram bins : "<< endl;
    cin>> histo_bin;
    

    unsigned int *data_host , * data_kernel;
//    float  *data_s;
// float bottom   [81] = {1, 2, 3, 5, 6, 7, 8, 9, 
//                        10,11,12,13,14,15,16,17,18,
//                        19,20,21,22,23,24,25,26,27,
//                        28,29,30,31,32,33,34,35,36,
//                        37,38,39,40,41,42,43,44,45,
//                        46,47,48,49,50,51,52,53,54,
//                        55,56,57,58,59,60,61,62,63,
//                        64,65,66,67,68,69,70,71,72,
//                        73,74,75,76,77,78,79,80,81
//                          };

    // cout << "width = " << img1.width();
    // cout << "height = " << img1.height();
     int img_width = img1.width();
     int img_height = img1.height();

    // int img_width = 9;
    // int img_height = 9;

    data_host = (unsigned int*) malloc(histo_bin * sizeof(unsigned int));
    data_kernel = (unsigned int*) malloc(histo_bin * sizeof(unsigned int));

 //   data_s = (float*) malloc(9 * sizeof(float));
    


    // initialize input array
        for (int j = 0; j < histo_bin; j++)
        {
            data_host[j] = 0;
            data_kernel[j] = 0;
        }


    double op_num = (double)((double)img_height * (double)img_width);


    //           HOST PART
    clock_t start = clock();
     histo_2D_sequential(img1.data(), data_host, img_width , img_height, histo_bin);
     //  convolution_2D_sequential(bottom, chosen_mask, data_host, img_width , img_height);

    clock_t stop = clock();
    double time_spent_host = (double)(stop - start) / CLOCKS_PER_SEC;
    double GFLOPS_host = op_num / time_spent_host;

 //           device Part 
     double GFLOPS_device_kernel;
    clock_t start_dev = clock();
     histo_2D_OnDevice(img1.data(), data_kernel, img_width, img_height,histo_bin, GFLOPS_device_kernel);
     //   convolution_2D_OnDevice(bottom, data_kernel, chosen_mask, img_width, img_height, GFLOPS_device_kernel);

    cudaDeviceSynchronize();
    clock_t stop_dev = clock();
    double time_spent_device = (double)(stop_dev - start_dev) / CLOCKS_PER_SEC;
    double GFLOPS_device_wrapper = op_num / time_spent_device;

  //convolution_2D_sequential(bottom, bottom_sobel, data_s, 3 , 3);


    cout << "Image Size :  " << img_height << " * " << img_width << endl;

               //  if(i >=1 && j >=1 )
                // cout <<"data img [ "<< i -1  <<"]["<<j  <<"] = " 
                //<< img1.data()[(i-1)*img_width + (j-2)]<<endl;
    //                      logical error checking
        for (int j = 0; j < histo_bin; j++)
        {
            if ( data_kernel[ j] != data_host[ j]  )
             {
                 cout << "Logical Error in values "<<endl;
                 cout << "data device ["<<j<<"] = " << data_kernel[j] 
                 << "   data host ["<<j << "] = " <<data_host[j] << endl;
                 //exit(EXIT_FAILURE);
             } 
        }

     CImg<float> img2(data_kernel, img_width, img_height, 1, 1);
     img2.save("cat_blur.jpeg");


    cout << " GFLOPS on host = " << GFLOPS_host/10e9 << endl;
    cout << " GFLOPS on device (wraaper) = " << GFLOPS_device_wrapper/10e9 << endl;
    cout << " GFLOPS on device (kernel) = " << GFLOPS_device_kernel/10e9 << endl;
    cout << " Speedup device(wrapper) vs host =  " << GFLOPS_device_wrapper/GFLOPS_host << endl;
    cout << " Speedup device(kernel) vs host =  " << GFLOPS_device_kernel/GFLOPS_host << endl;
    

    return 0;
}