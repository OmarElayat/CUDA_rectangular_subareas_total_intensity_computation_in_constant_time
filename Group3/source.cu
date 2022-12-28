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
#define Threadnum 512.0
#define S_Threadnum 4.0


//#define TILE_DIM 2

__global__ void sum_row_kernel_1(float* X, float* Y,float* S, int width,int height)
{
    __shared__ float T[(int)Threadnum];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col<width && row < height) {
    T[threadIdx.x] = X[row*width + col];
    
    //the code below performs iterative scan on T
    for (int stride = 1; stride < blockDim.x; stride *= 2){
    __syncthreads();
    float temp;
    if(threadIdx.x >= stride) {
    temp = T[threadIdx.x - stride];
    __syncthreads();
    if(threadIdx.x >= stride) {
    T[threadIdx.x] += temp;
    }
    }

    }
    __syncthreads();
    
    if(threadIdx.x == 0){S[blockIdx.y*gridDim.x+blockIdx.x] = T[(int)Threadnum-1];}

    Y[row*width + col] = T[threadIdx.x];
}
}

__global__ void sum_row_kernel_2(float* X, int width,int height)
{
    //__global__ void Kogge-stone_scan_kernel(float *X, float *Y, int InputSize)
    __shared__ float T[(int)S_Threadnum];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col<width && row < height) {
    T[threadIdx.x] = X[row*width + col];
    
    //the code below performs iterative scan on T
    for (int stride = 1; stride < blockDim.x; stride *= 2){
    __syncthreads();
    float temp;
    if(threadIdx.x >= stride) {
    temp = T[threadIdx.x - stride];
    __syncthreads();
    if(threadIdx.x >= stride) {
    T[threadIdx.x] += temp;
    }
    }

    }
    __syncthreads();

        X[row*width + col] = T[threadIdx.x];
    }
}

__global__ void sum_row_kernel_3(float* X, float* s, int width,int height)
{
    //__global__ void Kogge-stone_scan_kernel(float *X, float *Y, int InputSize)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col<width && row < height) {
    if(blockIdx.x >0){
        X[row*width + col] += s[(blockIdx.y*gridDim.x+blockIdx.x)-1];
    }
    }
    __syncthreads();
}


// __global__ void matTranspose_naive(float *A, float *B,int N,int h) {
//  /* Calculate global index for this thread */
//  int i = blockIdx.y * blockDim.y + threadIdx.y;
//  int j = blockIdx.x * blockDim.x + threadIdx.x;
//  /* Copy A[j][i] to B[i][j] */
//  if(j<N && i <h){
//  B[i *N + j] = A[j * N + i];
//  }
// }

__global__ void matTranspose_naive(const float* idata, float *odata,int width, int height)
{
   unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
   
   if (xIndex < width && yIndex < height)
   {
       unsigned int index_in  = xIndex + width * yIndex;
       unsigned int index_out = yIndex + height * xIndex;
       odata[index_out] = idata[index_in]; 
   }
}



void sat_OnDevice (float* mat, float* out_mat, int height, int width , double & gflops, double& op_num_heir)  {

    
    float *d_mat, *d_out_mat;
    int size = width * height*sizeof(float);
 
    float *S_r, *d_out_trans;
    int width_b = ceil(width/Threadnum);
    int height_b = ceil(height/Threadnum);

    
    cudaError_t err10 = cudaMalloc((void**) &d_out_trans, size);
    if (err10 != cudaSuccess) {
        printf("%s in %s at line %d\n",
        cudaGetErrorString(err10), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaError_t err4 = cudaMalloc((void**) &S_r, width_b*height*sizeof(float));
    if (err4!= cudaSuccess) {
        printf("%s in %s at line %d\n",
        cudaGetErrorString(err4), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaError_t err1 = cudaMalloc((void**) &d_mat, size);
    if (err1!= cudaSuccess) {
        printf("%s in %s at line %d\n",
        cudaGetErrorString(err1), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    float* d_out_mat_t;
    cudaError_t err19 = cudaMalloc((void**) &d_out_mat_t, size);
    if (err19!= cudaSuccess) {
        printf("%s in %s at line %d\n",
        cudaGetErrorString(err19), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy( d_mat, mat,  size, cudaMemcpyHostToDevice);

    cudaError_t err2 = cudaMalloc((void**) &d_out_mat, size);
    if (err2!= cudaSuccess) {
        printf("%s in %s at line %d\n",
        cudaGetErrorString(err2), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    //op_num_heir = (double)(((((double)log(width) * (double)width - ((double)width-1))+(((double)width/1024)*(double)log((double)width/1024)-((double)width/1024)-1) +(width- width/Threadnum))*height)+(((double)log(height) * (double)height - ((double)height-1))+(((double)height/1024)*log((double)height/1024)-((double)height/1024)-1) +((double)height- (double)height/Threadnum))*(double)width);
    
    clock_t start_dev = clock();

    dim3 dimGrid (ceil(width/Threadnum),height,1);
    dim3 dimBlock (Threadnum,1,1);

    sum_row_kernel_1<<<dimGrid, dimBlock>>>(d_mat, d_out_mat ,S_r, width, height);
    cudaDeviceSynchronize();

    dim3 dimGrid2 (1,height,1);
    dim3 dimBlock2 (S_Threadnum,1,1);
    sum_row_kernel_2<<<dimGrid2, dimBlock2>>>(S_r, ceil(width/Threadnum), height);
    cudaDeviceSynchronize();

    sum_row_kernel_3<<<dimGrid, dimBlock>>>(d_out_mat,S_r, width, height);
    cudaDeviceSynchronize();
    matTranspose_naive<<<dimGrid, dimBlock>>>(d_out_mat,d_out_trans,width,height);
    cudaDeviceSynchronize();

    dim3 dimGrid_c (ceil(height/Threadnum),width,1);
    sum_row_kernel_1<<<dimGrid_c, dimBlock>>>(d_out_trans, d_out_mat_t ,S_r, height, width);
    cudaDeviceSynchronize();

    dim3 dimGrid2_c (1,width,1);
    dim3 dimBlock2_c (ceil(height/S_Threadnum),1,1);
    sum_row_kernel_2<<<dimGrid2_c, dimBlock2_c>>>(S_r, ceil(height/Threadnum), width);
    cudaDeviceSynchronize();
    dim3 dimGrid3_c (ceil((height/Threadnum)),width,1);
    sum_row_kernel_3<<<dimGrid3_c, dimBlock>>>(d_out_mat_t,S_r, height,width);
    cudaDeviceSynchronize();
  
    matTranspose_naive<<<dimGrid, dimBlock>>>(d_out_mat_t,d_out_trans,height,width);
    cudaDeviceSynchronize();



    clock_t stop_dev = clock();
    double time_spent_device = (double)(stop_dev - start_dev) / CLOCKS_PER_SEC;
    gflops = op_num_heir / time_spent_device;

    
    cudaError_t err6 = cudaMemcpy(out_mat, d_out_trans, size, cudaMemcpyDeviceToHost);
    if (err6!= cudaSuccess) {
        printf("%s in %s at line %d\n",
        cudaGetErrorString(err6), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaFree(d_mat); 
    cudaFree(d_out_mat);
    cudaFree(d_out_trans);
    cudaFree(S_r);
}



void sat_sequential(float* mat, float* temp, int M, int N)
{
for (int i=0; i<N; i++)
	temp[i] = mat[i];

for (int i=1; i<M; i++) //col-wise sum
	for (int j=0; j<N; j++)
		temp[(i*N) + j] = mat[(i*N)+j] + temp[((i-1)*N)+j];

for (int i=0; i<M; i++)  //row-wise sum
	for (int j=1; j<N; j++)
		temp[(i*N)+j] += temp[(i*N)+(j-1)];
}


void rect_selector(int M, int N,int num_rect,int* tli, int* tlj, int* rbi,int* rbj){
    
for(int i =0;i<4*num_rect;i++){
    if(i%4 == 0) {
        cout << "tli_rect["<<i/4<<"]: ";
        cin >> tli[i/4];
    }else if(i%4 == 1){
        cout << "tlj_rect["<<i/4<<"]: ";
        cin >> tlj[i/4];
    }else if(i%4 == 2){
        cout << "rbi_rect["<<i/4<<"]: ";
        cin >> rbi[i/4];
    }else{
        cout << "rbj_rect["<<i/4<<"]: ";
        cin >> rbj[i/4];
    }
}
}

float total_intensity(float* temp,int M,int N, int tli, int tlj, int rbi,int rbj)
{
    
    float res = temp[(rbi*N)+rbj];
    if (tli > 0)
       res = res - temp[(tli-1)*N+rbj];
    if (tlj > 0)
       res = res - temp[(rbi*N)+tlj-1];
    if (tli > 0 && tlj > 0)
       res = res + temp[((tli-1)*N)+tlj-1];
	return res;
}


int main (){

    char  file[] = "test.jpeg";
    cout << "enter image file : " ;
    cin>> file;

     // float mat[60] = {31, 2, 4, 33,5,
	 // 		12,26,9,10,29,
	 // 		13,17,21,22,20,
	 // 		24,23,15,16,14,
     //        30,8,28,27,11,
     //        31, 2, 4, 33,5,
	 // 		12,26,9,10,29};
     //int img_width = 5;
     //int img_height = 7;
     //cout << "width = " << img_width ;
     //cout << "height = " <<img_height;


    CImg<float> img1("test.jpeg");
    float *data_host , *data_kernel;
    cout << "width = " << img1.width();
    cout << "height = " << img1.height();
    int img_width = img1.width();
    int img_height = img1.height();

    data_host = (float*) malloc(img_width * img_height * sizeof(float));
    data_kernel = (float*) malloc(img_width * img_height * sizeof(float));

    for(int i =0; i < img_height; i++){
        for(int j = 0; j<img_width; j++){
            data_host[i*img_height + j]=0;
        }
    }

    for(int i =0; i < img_height; i++){
        for(int j = 0; j<img_width; j++){
            data_kernel[i*img_height + j]=0;
        }
    }

    double op_num = (double)((double)img_height * (double)img_width * 2 - (double)img_height - (double)img_width);

/////////////////////////Rectangles initialization///////////////////////


int num_rect;

cout<< "num of rect";
cin >> num_rect;

int* tli,*tlj,*rbi,*rbj;
float* rect_intensity_seq,*rect_intensity_kern;
tli = (int*)malloc(num_rect*sizeof(int));
tlj = (int*)malloc(num_rect*sizeof(int));
rbi = (int*)malloc(num_rect*sizeof(int));
rbj = (int*)malloc(num_rect*sizeof(int));
rect_intensity_seq = (float*)malloc(num_rect*sizeof(float));
rect_intensity_kern=(float*)malloc(num_rect*sizeof(float));

rect_selector(img_height,img_width,num_rect,tli, tlj, rbi, rbj);

/////////////////////////////////////////////////////////////////////////




    //           HOST PART
    clock_t start = clock();
    sat_sequential(img1.data(), data_host,img_height,img_width);
    //sat_sequential(mat, data_host, img_height,img_width);
    for(int i =0;i<num_rect;i++){
        //rect_intensity_seq[i] = total_intensity(mat, img_height, img_width,tli[i], tlj[i], rbi[i], rbj[i]);
        rect_intensity_seq[i] = total_intensity(img1.data(), img_height, img_width,tli[i], tlj[i], rbi[i], rbj[i]);
        cout << "\nQuery_sequential[" << i <<"]: " << rect_intensity_seq[i]<<endl;
    }
    clock_t stop = clock();
    //total_intensity(data_host);
    

    double time_spent_host = (double)(stop - start) / CLOCKS_PER_SEC;
    double GFLOPS_host = op_num / time_spent_host;

    float* out_mat_s= (float*) malloc(ceil(img_width/Threadnum) * img_height * sizeof(float));
    float* out_mat_trans_1= (float*) malloc(img_width * img_height * sizeof(float));
 //           device Part 
    double GFLOPS_device_kernel;
    double op_num_dev;
    clock_t start_dev = clock();
    sat_OnDevice(img1.data(), data_kernel,img_height,img_width,GFLOPS_device_kernel,op_num);
    //sat_OnDevice(mat, data_kernel,img_height,img_width,GFLOPS_device_kernel,op_num);
    cudaDeviceSynchronize();
    for(int i =0;i<num_rect;i++){
        rect_intensity_kern[i] = total_intensity(data_kernel, img_height, img_width,tli[i], tlj[i], rbi[i], rbj[i]);
        cout << "\nQuery_kernel[" << i <<"]: " << rect_intensity_kern[i] << endl;
    }
    clock_t stop_dev = clock();
    
     double time_spent_device = (double)(stop_dev - start_dev) / CLOCKS_PER_SEC;
    double GFLOPS_device_wrapper = op_num / time_spent_device;


        //                  logical error checking
     int k =0;
     for (int i = 0; i < img_height; i++)
     {
         for (int j = 0; j < img_width; j++)
         {
             if ( abs(data_kernel[i*img_width + j] - data_host[i*img_width + j]) > (float) 10.1)
              {
                  cout << "Logical Error in values "<<endl;
                  cout << "data device [ "<< i*img_width + j <<"] = " << data_kernel[i*img_width + j] << "   data host [ " << i*img_width + j << "] = " <<data_host[i*img_width + j] << endl;
                  exit(EXIT_FAILURE);
                  k++;
              } 
         }
     }
     cout << "errors equals" << k << endl;
    cout << " GFLOPS on host = " << GFLOPS_host/10e9 << endl;
    cout << " GFLOPS on device (wraaper) = " << GFLOPS_device_wrapper/10e9 << endl;
    cout << " GFLOPS on device (kernel) = " << GFLOPS_device_kernel/10e9 << endl;
    cout << " Speedup device(wrapper) vs host =  " << GFLOPS_device_wrapper/GFLOPS_host << endl;
    cout << " Speedup device(kernel) vs host =  " << GFLOPS_device_kernel/GFLOPS_host << endl;
    

float* x = img1.data();
for(int i =0; i < 20; i++){
        for(int j = 0; j<20; j++){
            printf("out_device[%d,%d] = %f \t",i,j,x[i*img_width + j]);
        }
        printf("\n");
    }
for(int i =0; i < 20; i++){
        for(int j = 0; j<20; j++){
            printf("data_host[%d,%d] = %f \t",i,j,data_host[i*img_width + j]);
        }
        printf("\n");
    }
    
for(int i =0; i < 20; i++){
        for(int j = 0; j<20; j++){
            printf("data_kernel[%d,%d] = %f \t",i,j,data_kernel[i*img_width + j]);
        }
        printf("\n");
    }
    return 0;
}
