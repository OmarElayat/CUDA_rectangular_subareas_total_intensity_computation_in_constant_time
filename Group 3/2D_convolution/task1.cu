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

#define MASK_WIDTH 3

    __constant__ float M[MASK_WIDTH * MASK_WIDTH];
    // cudaMemcpyToSymbol(M, mask, MASK_WIDTH* MASK_WIDTH*sizeof(float));

__global__ void convolution_2D_kernel(float* N, float* P,int width , int height)
{


    __shared__ float N_ds[TILE_WIDTH * TILE_WIDTH];

 // Calculate the row index of the P element and M
    int Row = blockIdx.y*blockDim.y + threadIdx.y;
 // Calculate the column index of P and N
    int Col = blockIdx.x*blockDim.x + threadIdx.x;

    if (Row < height && Col < width)
    {
    int This_tile_start_point_x = blockIdx.x * blockDim.x;
    int Next_tile_start_point_x = (blockIdx.x + 1) * blockDim.x;

    int This_tile_start_point_y = blockIdx.y * blockDim.y;
    int Next_tile_start_point_y = (blockIdx.y + 1) * blockDim.y;

    int x_start_point = Col - (MASK_WIDTH/2);
    int y_start_point = Row - (MASK_WIDTH/2);

    N_ds[threadIdx.y * blockDim.x + threadIdx.x] = N[Row * width + Col];

    float Pvalue = 0;
    for (int i = 0; i < MASK_WIDTH; i ++) {
        for (int j = 0; j < MASK_WIDTH; j ++) {
            int x_index = x_start_point + j;
            int y_index = y_start_point + i;
            if (x_index >= 0 && x_index < width && y_index >= 0 && y_index < height)
            {
                 if ((x_index >= This_tile_start_point_x) && (x_index < Next_tile_start_point_x) 
                    && (y_index >= This_tile_start_point_y) && (y_index < Next_tile_start_point_y)
                    && (x_index < width) && (y_index < height)) {

                    Pvalue += N_ds[(threadIdx.y+i-(MASK_WIDTH/2)) * TILE_WIDTH + (threadIdx.x+j-(MASK_WIDTH/2))]* M[i * MASK_WIDTH + j];
                 }else {
                    Pvalue += N[y_index * width + x_index] * M[i * MASK_WIDTH + j];
                 }
            }else {
                    Pvalue += N[Row * width + Col] * M[i * MASK_WIDTH + j];                                  // replicating boundary cells - Assuming mask size is 3
            }
        }
    }
    if (Pvalue < 0 )
    {
        P[Row * width + Col] = 0;
    }else if (Pvalue > 255){
         P[Row * width + Col] = 255;
    }else{
        P[Row * width + Col] = Pvalue;
    }
}
}

void convolution_2D_OnDevice (float* N, float* P, float * mask, int width , int height , double & gflops)  {

    
    // __constant__ float M[MASK_WIDTH * MASK_WIDTH];
    // cudaMemcpyToSymbol(M, mask, MASK_WIDTH* MASK_WIDTH*sizeof(float));

    float *d_N, *d_P;
    int size = width * height * sizeof(float);
    cudaMemcpyToSymbol(M, mask, MASK_WIDTH* MASK_WIDTH*sizeof(float));


    cudaError_t err1 = cudaMalloc((void**) &d_N, size);

    if (err1!= cudaSuccess) {
        printf("%s in %s at line %d\n",
        cudaGetErrorString(err1), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy( d_N, N,  size, cudaMemcpyHostToDevice);

    cudaError_t err2 = cudaMalloc((void**) &d_P, size);
    if (err2!= cudaSuccess) {
        printf("%s in %s at line %d\n",
        cudaGetErrorString(err2), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    //cout << "d_n" << d_N[0]<<endl;

    dim3 dimGrid (ceil(width/8.0), ceil(height/8.0),1);
    dim3 dimBlock (8,8,1);

    double op_num = (double)((double)height * (double)width * (double) MASK_WIDTH * (double) MASK_WIDTH * 2);
    clock_t start_dev = clock();
    convolution_2D_kernel<<<dimGrid, dimBlock>>>(d_N,  d_P, width, height);
    cudaDeviceSynchronize();
    clock_t stop_dev = clock();
    double time_spent_device = (double)(stop_dev - start_dev) / CLOCKS_PER_SEC;
    gflops = op_num / time_spent_device;

    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

    cudaFree(d_N); 
    cudaFree(d_P);
}



void convolution_2D_sequential(float* N, float * M, float* P, int width , int height) {

// find center position of kernel (half of kernel size)
int kCenterX = MASK_WIDTH / 2;
int kCenterY = MASK_WIDTH / 2;

for (int i = 0; i < height; ++i)              // rows
{
    for (int j = 0; j < width; ++j)          // columns
    {
        for (int m = 0; m < MASK_WIDTH; ++m)     // kernel rows
        {
          int ii = i + (m - kCenterY);
            for (int n = 0; n < MASK_WIDTH; ++n) // kernel columns
            {
                // index of input signal, used for checking boundary
                
                int jj = j + (n - kCenterX);
        //    cout << " jj = " << jj << "  ii = "<<ii<<" i = "<<i<<"  j = "<<j;
                // ignore input samples which are out of bound
                if (ii >= 0 && ii < height && jj >= 0 && jj < width){
                    P[i * width + j] += N[ii* width + jj] * M[m * MASK_WIDTH + n];
                }
                else{
                    P[i * width + j] += N[i * width + j]* M[m * MASK_WIDTH + n];
                }

                 //   cout << "     written !" << "p["<<i * width + j<<"]" 
                 //   << "plus " <<N[ii* width + jj] * M[m][n] <<endl;
                // } else {
                //   cout<<endl;
                // }
            }
        }
       if (P[i * width + j] < 0 ){
            P[i * width + j] = 0;
        }else if (P[i * width + j] > 255){
            P[i * width + j] = 255;
        }
    }
}
}


void copy_mask(float *M , float * x ){
    for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; ++i)
    {
        M[i] = x[i];
    }
}

int main (){

    int x;
    char  file[] = "Cat0.jpeg";
    cout << "enter image file : " ;
    cin>> file;
    //char* filename = "Cat0.jpeg";
    CImg<float> img1(file);
   // CImg<float> img1("Cat0.jpeg");

    cout << "enter operation code : "<< endl <<
    "1 - blur  " << endl <<
    "2 - emboss  " <<endl <<
    "3 - outline  " << endl <<
    "4 - sharpen  " << endl <<
    "5 - left sobel  " << endl <<
    "6 - right sobel  " << endl <<
    "7 - top sobel  " << endl <<
    "8 - bottom sobel  " << endl <<
    "9 - identity  " << endl;
    cin>> x;
    
    float chosen_mask [9];

// masks are flipped

    float blur_mask    [9] = {0.0625,0.125,0.0625,
                              0.125 ,0.25 ,0.125, 
                              0.0625,0.125,0.0625
                             };
    float emboss_mask    [9] = {2 ,1, 0,
                                1 ,1 , -1, 
                                 0 ,-1 ,-2
                               };
    float outline_mask    [9] = {-1 ,-1, -1,
                                 -1 ,8 , -1, 
                                 -1 ,-1 ,-1
                                };
    float sharpen_mask    [9] = {0, -1, 0,
                                 -1, 5, -1,
                                  0, -1, 0
                                };
    float left_sobel      [9] = {-1, 0, 1,
                                 -2, 0, 2,
                                 -1, 0, 1
                                };                                
    float right_sobel      [9] = {1, 0, -1,
                                  2, 0, -2,
                                  1, 0, -1
                                };
    float top_sobel      [9] = {-1, -2, -1,
                                0, 0, 0,
                                1, 2, 1
                                }; 
    float bottom_sobel      [9] = {1, 2, 1,
                                    0, 0, 0,
                                    -1, -2,-1
                                  };
        float identity      [9] = {0, 0, 0,
                                    0, 1, 0,
                                    0, 0, 0
                                  }; 

    switch (x)  {
    case 1: copy_mask(chosen_mask,blur_mask); break;
    case 2: copy_mask(chosen_mask,emboss_mask) ; break;
    case 3: copy_mask(chosen_mask,outline_mask) ; break;
    case 4: copy_mask(chosen_mask,sharpen_mask) ; break;
    case 5: copy_mask(chosen_mask,left_sobel) ; break;
    case 6: copy_mask(chosen_mask,right_sobel) ; break;
    case 7: copy_mask(chosen_mask,top_sobel) ; break;
    case 8: copy_mask(chosen_mask,bottom_sobel) ; break;
    default: copy_mask(chosen_mask,identity);
    } 

    float *data_host , * data_kernel;
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

    data_host = (float*) malloc(img_width * img_height * sizeof(float));
    data_kernel = (float*) malloc(img_width * img_height * sizeof(float));

 //   data_s = (float*) malloc(9 * sizeof(float));
    


    // initialize input array
    for (int i = 0; i < img_height; i++)
    {
        for (int j = 0; j < img_width; j++)
        {
            data_host[i* img_width + j] = 0;
            data_kernel[i* img_width + j] = 0;
            //cout << "Pin_M[" << i << "]" << "[" << j << "] = "  <<  data[i* img_width + j] << "    " ;
        }
        //cout <<endl;
    }


    double op_num = (double)((double)img_height * (double)img_width * (double) MASK_WIDTH * (double) MASK_WIDTH * 2);


    //           HOST PART
    clock_t start = clock();
     convolution_2D_sequential(img1.data(), chosen_mask, data_host, img_width , img_height);
     //  convolution_2D_sequential(bottom, chosen_mask, data_host, img_width , img_height);

    clock_t stop = clock();
    double time_spent_host = (double)(stop - start) / CLOCKS_PER_SEC;
    double GFLOPS_host = op_num / time_spent_host;

 //           device Part 
     double GFLOPS_device_kernel;
    clock_t start_dev = clock();
     convolution_2D_OnDevice(img1.data(), data_kernel, chosen_mask, img_width, img_height, GFLOPS_device_kernel);
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
    for (int i = 0; i < img_height; i++)
    {
        for (int j = 0; j < img_width; j++)
        {
            if ( abs(data_kernel[i*img_width + j] - data_host[i*img_width + j]) > (float) 3.1)
             {
                 cout << "Logical Error in values "<<endl;
                 cout << "data device [ "<< i <<"]["<<j<<"] = " << data_kernel[i*img_width + j] 
                 << "   data host [ " << i<<"]["<<j << "] = " <<data_host[i*img_width + j] << endl;
                 //exit(EXIT_FAILURE);
             } 
        }
    }


    //     for (int i = 0; i < img_height; i++)
    // {
    //     for (int j = 0; j < img_width; j++)
    //     {
    //         cout << "kernel[" << i << "]" << "[" << j << "] = "  <<  (float) data_kernel[i*img_width + j] << "    " ;
    //     }
    //     cout <<endl;
    // }
    //         cout <<endl;

    // for (int i = 0; i < img_height; i++)
    // {
    //     for (int j = 0; j < img_width; j++)
    //     {
    //         cout << "host[" << i << "]" << "[" << j << "] = "  <<  (float) data_host[i*img_width + j] << "    " ;
    //     }
    //     cout <<endl;
    // }
    // cout <<endl;

     CImg<float> img2(data_kernel, img_width, img_height, 1, 1);
     img2.save("cat_blur.jpeg");


    cout << " GFLOPS on host = " << GFLOPS_host/10e9 << endl;
    cout << " GFLOPS on device (wraaper) = " << GFLOPS_device_wrapper/10e9 << endl;
    cout << " GFLOPS on device (kernel) = " << GFLOPS_device_kernel/10e9 << endl;
    cout << " Speedup device(wrapper) vs host =  " << GFLOPS_device_wrapper/GFLOPS_host << endl;
    cout << " Speedup device(kernel) vs host =  " << GFLOPS_device_kernel/GFLOPS_host << endl;
    

    return 0;
}