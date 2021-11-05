#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"

#define MAX 65535



__global__ void vecAdd(float* arr_A,float* arr_B, float* arr_C, int SIZE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < SIZE)
	{
		arr_C[tid] = arr_A[tid] + arr_B[tid];
		tid = tid + blockDim.x * gridDim.x;
	}
}



void main(int argc, char **argv)
{
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time;

    const int ARRAY_SIZE = atoi(argv[1]);
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	
	float h_a[MAX];
	float h_b[MAX];
	float h_c[MAX];

	float *dev_a, *dev_b, *dev_c;

	int i;
	float f = 0.0;

	for (i = 0; i < ARRAY_SIZE; i++)
	{
		h_a[i] = f * f;
		h_b[i] = f * f * f;
		f++;
	}

	cudaMalloc((void**)&dev_a, ARRAY_BYTES);
	cudaMalloc((void**)&dev_b, ARRAY_BYTES);
	cudaMalloc((void**)&dev_c, ARRAY_BYTES);

	cudaMemcpy(dev_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, h_b, ARRAY_BYTES, cudaMemcpyHostToDevice);

	cudaEventRecord(start,0);
	vecAdd <<< (ARRAY_SIZE+127)/128,128 >>>(dev_a,dev_b,dev_c,ARRAY_SIZE);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(h_c, dev_c, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	/*for (i = 0; i < ARRAY_SIZE; i++)
	{
		printf("%f + %f = %f \n", h_a[i], h_b[i], h_c[i]);
	}*/
	
	printf("Time taken for kernel execution %f\n", time);

}



