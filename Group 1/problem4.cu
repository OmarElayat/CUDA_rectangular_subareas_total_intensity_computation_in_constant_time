#include <stdlib.h>
#include<stdio.h>
#include<cmath>
#include<time.h>
/*Write a full CUDA program to perform matrix addition on square matrices such that each thread is responsible for
computing one column of the output (sum) matrix. The size of the matrices and their values should be randomly
selected.
*/
__global__ void matAdd(float* a, float* b, float* c , int N){

int i = threadIdx.x + blockIdx.x * blockDim.x;
if(i < N ){
for(int j = 0; j<N; j++){
    c[i+(j*N)] = a[i+(j*N)] + b[i+(j*N)];
}  
}
}
int main(){
srand(time(0));
int cols = rand();
int rows = cols;
printf("cols: %d \t rows: %d",cols,rows);
float *a, *b, *c, *d_a, *d_b, *d_c;
a = (float *) malloc(rows*cols*sizeof(float));
b = (float *) malloc(rows*cols*sizeof(float));
c = (float *) malloc(rows*cols*sizeof(float));

for(int i =0; i <(cols*rows);i++){
    a[i] = rand();
    b[i] = rand();
}

// for(int i =0; i < rows; i++){
//     for(int j = 0; j<cols; j++){
//         a[j+i*cols] = j+i*cols;
//         b[j+i*cols] = 2*(j+i*cols);
//     }
// }

cudaMalloc ((void **) &d_a, rows*cols*sizeof(float));
cudaMalloc ((void **) &d_b, rows*cols*sizeof(float));
cudaMalloc ((void **) &d_c, rows*cols*sizeof(float));
cudaMemcpy(d_a,a,rows*cols*sizeof(float),cudaMemcpyHostToDevice);
cudaMemcpy(d_b,b,rows*cols*sizeof(float),cudaMemcpyHostToDevice);
dim3 dimGrid(ceil(cols/256.0),1);
dim3 dimBlock(256.0,1);

matAdd<<<dimGrid,dimBlock>>>(d_a,d_b,d_c,cols);
cudaMemcpy(c,d_c,rows*cols*sizeof(float),cudaMemcpyDeviceToHost);

printf("\n");
for(int i =0; i < rows; i++){
    for(int j = 0; j<cols; j++){
        printf("a[%d,%d] = %f \t",i,j,a[j+i*cols]);
    }
    printf("\n");
}
printf("\n");
for(int i =0; i < rows; i++){
    for(int j = 0; j<cols; j++){
        printf("b[%d,%d] = %f \t",i,j,b[j+i*cols]);
    }
    printf("\n");
}
printf("\n");
for(int i =0; i < rows; i++){
    for(int j = 0; j<cols; j++){
        printf("c[%d,%d] = %f \t",i,j,c[j+i*cols]);
    }
    printf("\n");
}

cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
free(a);
free(b);
free(c);
}