#include <stdlib.h>
#include<stdio.h>
#include<cmath>
#include<time.h>
/*Write a full CUDA program to perform vector addition such that each thread is responsible for computing four
adjacent elements in the output vector instead of one. The vectors size as well as data should be randomly generated
(Hint: Use C rand and srand functions). The program should print the vectors size, both input vectors, and the
output vector at the end.
What is the maximum size of the vectors that can be used if the kernel is launched with a single block?
*/
__global__ void vecAdd(float* d_a, float* d_b, float* d_c,int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < ceil(N/4.0)){
        d_c[4*i] = d_a[4*i]+ d_b[4*i];
        d_c[(4*i)+1] = d_a[(4*i)+1] + d_b[(4*i)+1];
        d_c[(4*i)+2] = d_a[(4*i)+2] + d_b[(4*i)+2];
        d_c[(4*i)+3] = d_a[(4*i)+3] + d_b[(4*i)+3];
    }
}
int main(){
    srand(time(0));
    int N = rand();
    float *a, *b, *c, *d_a, *d_b, *d_c;
    a = (float *) malloc(N * sizeof(float));
    b = (float *) malloc(N * sizeof(float));
    c = (float *) malloc(N * sizeof(float));
    for(int i = 0; i<N; i++){
        a[i] = rand();
        b[i] = rand();
    }
    cudaMalloc ((void **) &d_a, N*sizeof(float)); 
    cudaMemcpy (d_a,a,N * sizeof(float),cudaMemcpyHostToDevice);
    cudaMalloc ((void **) &d_b, N*sizeof(float)); 
    cudaMemcpy (d_b,b,N * sizeof(float),cudaMemcpyHostToDevice);
    
    cudaMalloc ((void **) &d_c, N*sizeof(float)); 
    
    vecAdd<<<ceil(N/(256.0*4.0)),256>>>(d_a,d_b,d_c,N);

    cudaMemcpy(c,d_c,N * sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++){
        printf("a[%d] = %f \n",i, a[i]);
        printf("b[%d] = %f \n",i, b[i]);
        printf("c[%d] = %f \n",i, c[i]);
    }
    printf("Vectors size: %d", N);
    cudaFree(d_b);
    cudaFree(d_a);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

}