#include "cuda_runtime.h"
#include <stdio.h>
#include "device_launch_parameters.h"


__global__ void VecAdd(float *a ,float *b , float*c , int N)
{
	int idx = threadIdx.x;
	if (idx < N)
	{
		c[idx] = a[idx] + b[idx];
	}
}


void main(void)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	const int ARRAY_SIZE = 1024;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	
	float h_arrA[ARRAY_SIZE];
	float h_arrB[ARRAY_SIZE];
	float h_arrC[ARRAY_SIZE];
	float* d_arrA;
	float* d_arrB;
	float* d_arrC;

	int i;
	float f = 0.0;

	//initializing the host arrays
	for (i = 0; i < ARRAY_SIZE; i++)
	{
		h_arrA[i] = f;
		h_arrB[i] = f * f;
		f = f + 1.0;
	}
	//initializing the device arrays
	
	cudaMalloc((void**)&d_arrA, ARRAY_BYTES);
	cudaMalloc((void**)&d_arrB, ARRAY_BYTES);
	cudaMalloc((void**)&d_arrC, ARRAY_BYTES);

	cudaMemcpy(d_arrA, h_arrA, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_arrB, h_arrB, ARRAY_BYTES, cudaMemcpyHostToDevice);
	
	cudaEventRecord(start, 0);
	VecAdd <<<1 ,ARRAY_SIZE>>>(d_arrA,d_arrB,d_arrC,ARRAY_SIZE);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(h_arrC, d_arrC, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	for (i = 0; i < ARRAY_SIZE; i++)
	{
		printf("%f + %f = %f \n", h_arrA[i], h_arrB[i], h_arrC[i]);
	}
	printf("The Time taken by GPU is : %f", time);
	cudaFree(d_arrA);
	cudaFree(d_arrB);
	cudaFree(d_arrC);
}















