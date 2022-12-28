#include<stdio.h>
#include<cmath>
#include<ctime>

/*
Write a full program to randomly generate a grayscale picture by generating a 2D array of integers of size 1000x800
randomly initialized to values ranging between 0 and 255. 

Your program should then use a CUDA kernel with a 2D
grid and 2D blocks to multiply each pixel of the picture by 3 (trimming the resulting value to 255 if it exceeds that
value). Each block should have 16x16 threads and each thread should be responsible for a single pixel.
How many blocks in each dimension will we have? How many threads in total will be in the grid? 
*/
__global__ void greyScale(unsigned char* in,unsigned char* out, int h, int w){
    int cols = (blockIdx.x * blockDim.x + threadIdx.x);
    int rows = (blockIdx.y * blockDim.y + threadIdx.y);
    if(cols < w && rows < h){
      if(in[(cols + rows*w)] <= 85){
        out[(cols + rows*w)] = in[(cols + rows*w)]*3;
      }else{
        out[(cols + rows*w)] = 255;
      }
    }
}
int main(){
srand(time(0));
int rows = 1000;
int cols = 800;
unsigned char *in, *out, *d_in, *d_out;
in = (unsigned char *) malloc(rows*cols*sizeof(char));
out = (unsigned char *) malloc(rows*cols*sizeof(char));

for(int i =0; i < rows; i++){
    for(int j = 0; j<cols; j++){
        in[j+i*cols] = rand()%256;
    }
}
dim3 dimGrid(ceil(cols/16.0),ceil(rows/16.0),1);
dim3 dimBlock(16.0,16.0,1);

cudaMalloc ((unsigned char **) &d_in, rows*cols*sizeof(char));
cudaMalloc ((unsigned char **) &d_out, rows*cols*sizeof(char));

cudaMemcpy(d_in,in,rows*cols*sizeof(char),cudaMemcpyHostToDevice);
greyScale<<<dimGrid,dimBlock>>>(d_in,d_out,rows,cols);
cudaMemcpy(out,d_out,rows*cols*sizeof(char),cudaMemcpyDeviceToHost);
    
printf("\n");
for(int i =0; i < rows; i++){
    for(int j = 0; j<cols; j++){
        printf("in[%d,%d] = %d \t",i,j,in[j+i*cols]);
    }
    printf("\n");
}
printf("\n");
for(int i =0; i < rows; i++){
    for(int j = 0; j<cols; j++){
        printf("out[%d,%d] = %d \t",i,j,out[j+i*cols]);
    }
    printf("\n");
}

free(in);
free(out);
cudaFree(d_in);
cudaFree(d_out);
}