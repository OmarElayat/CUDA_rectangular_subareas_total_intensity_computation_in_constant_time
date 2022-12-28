
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <assert.h>

using namespace std;

#define width_N_M 1200
#define height_N 1300
#define height_M 1500
#define MAX_FLOAT_NUM 100.0

#define TILE_DIM 32.0
#define TILE_DIM_INT int(TILE_DIM)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#  undef  assert
#  define assert(arg)
#endif

__global__ void MatrixMultiplykernel(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols) {
    int CCols = BCols;
    int CRows = ARows;
    float CValue = 0;
    
    int Row = blockIdx.y*TILE_DIM_INT + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM_INT + threadIdx.x;
    __shared__ float As[TILE_DIM_INT][TILE_DIM_INT];
    __shared__ float Bs[TILE_DIM_INT][TILE_DIM_INT];
         
    for (int k = 0; k < ACols/TILE_DIM ; k++) {
            //(TILE_DIM + ACols - 1)
        if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows){
            As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM_INT + threadIdx.x];
        }
        else{
            As[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols){
            Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM_INT  + threadIdx.y)*BCols + Col];
        }
        else{
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int n = 0; n < TILE_DIM; ++n){
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (Row < CRows && Col < CCols) C[(Row*CCols)+Col]=CValue;
    //crows = arows ccols = bcols
}

/*__global__ void MatrixMultiplykernel(float* M, float* N, float* P, int Width, int heightN, int heightM)
{
 // Calculate the row index of the P element and M
    int Row = blockIdx.y*blockDim.y + threadIdx.y;
 // Calculate the column index of P and N
    int Col = blockIdx.x*blockDim.x + threadIdx.x;
    if (Col < heightN && Row < heightM)
    {
    
        float Pvalue = 0;
    // Each thread computes one element of the block sub-matrix
       for (int k = 0; k < Width; ++k)
            Pvalue += M[Row*Width+k] * N[k*heightN+Col];

        P[Row*heightN+Col] = Pvalue;
    }
}
*/

void MatrixMulOnHost (float * M, float * N, float * P, int width, int heightN, int heightM) {
    for (int i = 0; i < heightM; ++i){
        for (int j = 0; j < heightN; ++j) {
            float sum = 0;
            for (int k = 0; k < width; ++k) {
                float a = M[i * width + k];
                float b = N[k * heightN + j];
                sum += a * b;
            }
            P[i*heightN+j]=sum;
        }
    }
}

void MatrixMultiplyOnDevice (float * M, float * N, float * P, int width, int heightN, int heightM , double & gflops)  {

    int sizeN = width * heightN * sizeof(float);
    int sizeM = width * heightM * sizeof(float);
    int sizeout = heightN * heightM * sizeof(float);

    float *d_Pout, *d_PinM , *d_PinN;

    cudaError_t err1 = cudaMalloc((void**) &d_PinM, sizeM);

    if (err1!= cudaSuccess) {
        printf("%s in %s at line %d\n",
        cudaGetErrorString(err1), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy( d_PinM, M,  sizeM, cudaMemcpyHostToDevice);

    cudaError_t err2 = cudaMalloc((void**) &d_PinN, sizeN);
    if (err2!= cudaSuccess) {
        printf("%s in %s at line %d\n",
        cudaGetErrorString(err2), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }


    cudaMemcpy( d_PinN, N,  sizeN, cudaMemcpyHostToDevice);

    cudaError_t err3 = cudaMalloc((void**) &d_Pout, sizeout);
    if (err3!= cudaSuccess) {
        printf("%s in %s at line %d\n",
        cudaGetErrorString(err3), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }


    dim3 dimGrid (ceil(heightN/TILE_DIM), ceil(heightM/TILE_DIM),1);
    dim3 dimBlock (TILE_DIM,TILE_DIM,1);

    double op_num = (double)((double)heightM * (double)heightN * (2 * (double)width - 1 ));
    clock_t start_dev = clock();
    //MatrixMultiplykernel<<<dimGrid, dimBlock>>>(d_PinM,  d_PinN, d_Pout,width, heightN , heightM);
    MatrixMultiplykernel<<<dimGrid, dimBlock>>>(d_PinM, d_PinN, d_Pout, heightM , width, width, heightN);
    cudaDeviceSynchronize();
    clock_t stop_dev = clock();
    double time_spent_device = (double)(stop_dev - start_dev) / CLOCKS_PER_SEC;
    gflops = op_num / time_spent_device;

    cudaMemcpy(P, d_Pout, sizeout, cudaMemcpyDeviceToHost);

    cudaFree(d_PinN); 
    cudaFree(d_PinM);
    cudaFree(d_Pout);
}

int main (){

    float *Pin_N,*Pin_M,*Pout, *pout_device;
    Pin_N = (float*) malloc(width_N_M * height_N * sizeof(float));
    Pin_M = (float*) malloc(width_N_M * height_M * sizeof(float));
    Pout = (float*) malloc(height_M * height_N * sizeof(float));
    pout_device = (float*) malloc(height_M * height_N * sizeof(float));
    for (int i = 0; i < height_M * width_N_M; i++)
    {
       // Pin_M[i] =  (float) (rand() %((float)100.0));
      Pin_M[i] = static_cast <float> (rand()) / 
                 (static_cast <float> (RAND_MAX/MAX_FLOAT_NUM));
    }
    for (int i = 0; i < height_N * width_N_M; i++)
    {
       // Pin_N[i] =  (float) rand()%((float)100.0) ;
        Pin_N[i] = static_cast <float> (rand()) / 
                 (static_cast <float> (RAND_MAX/MAX_FLOAT_NUM));
    }

    double op_num = (double)((double)height_M * (double)height_N * (2 * (double)width_N_M - 1 ));
    //           HOST PART
    clock_t start = clock();
    MatrixMulOnHost(Pin_M, Pin_N, Pout, width_N_M, height_N , height_M);
    clock_t stop = clock();
//    Mat_Multiply(Pin_N,Pin_M, width_N_M, height);
    double time_spent_host = (double)(stop - start) / CLOCKS_PER_SEC;
    double GFLOPS_host = op_num / time_spent_host;


    //         device Part 
     double GFLOPS_device_kernel;
    clock_t start_dev = clock();
    MatrixMultiplyOnDevice(Pin_M, Pin_N, pout_device, width_N_M, height_N , height_M, GFLOPS_device_kernel);
    cudaDeviceSynchronize();
    clock_t stop_dev = clock();
//    Mat_Multiply(Pin_N,Pin_M, width_N_M, height);
    double time_spent_device = (double)(stop_dev - start_dev) / CLOCKS_PER_SEC;
    double GFLOPS_device_wrapper = op_num / time_spent_device;


    cout << "vectors size:  M " << height_M << " * " << width_N_M << endl;
    cout << "vectors size:  N " << width_N_M << " * " <<  height_N<< endl;
    cout << "Result vector size: " << height_M << " * " <<  height_N<< endl;

    //                      logical error checking
    for (int i = 0; i < height_M; i++)
    {
        for (int j = 0; j < height_N; j++)
        {
            if ( pout_device[i*height_N + j] - Pout[i*height_N + j] > (float) 10.1)
             {
                 cout << "Logical Error in values "<<endl;
                 cout << "pout device = " << pout_device[i*height_N + j] << "   pout = " << Pout[i*height_N + j] << endl;
                 exit(EXIT_FAILURE);
             } 
        }
    }
    cout <<endl;
    cout << " GFLOPS on host = " << GFLOPS_host/10e9 << endl;
    cout << " GFLOPS on device (wraaper) = " << GFLOPS_device_wrapper/10e9 << endl;
    cout << " GFLOPS on device (kernel) = " << GFLOPS_device_kernel/10e9 << endl;
    cout << " Speedup device(wrapper) vs host =  " << GFLOPS_device_wrapper/GFLOPS_host << endl;
    cout << " Speedup device(kernel) vs host =  " << GFLOPS_device_kernel/GFLOPS_host << endl;
/*
    printf("\n");
    for(int i =0; i < height_M; i++){
        for(int j = 0; j<height_N; j++){
            printf("out_device[%d,%d] = %f \t",i,j,pout_device[i*height_N + j]);
        }
        printf("\n");
    }
    
    printf("\n");
    for(int i =0; i < height_M; i++){
        for(int j = 0; j<height_N; j++){
            printf("out[%d,%d] = %f \t",i,j,Pout[i*height_N + j]);
        }
        printf("\n");
    }
*/
    return 0;
}