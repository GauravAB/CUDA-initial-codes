#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


__global__ void vecAdd(float *a, float *b, float *c, int N)
{
	int idx = threadIdx.x;
	if (idx < N)
	{
		c[idx] = a[idx] + b[idx];
	}

} 


void main(void)
{
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float time;
	
	const int ARRAY_SIZE = 1000;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	float h_a[ARRAY_SIZE];
	float h_b[ARRAY_SIZE];
	float h_c[ARRAY_SIZE];
	float *dev_a;
	float *dev_b;
	float *dev_c; 
	int i;
	float f = 0.0;

	for (i = 0; i < ARRAY_SIZE; i++)
	{
		h_a[i] = f;
		h_b[i] = f * f;
		f++;
	}

	cudaMalloc((void**)&dev_a, ARRAY_BYTES);
	cudaMalloc((void**)&dev_b, ARRAY_BYTES);
	cudaMalloc((void**)&dev_c, ARRAY_BYTES);

	cudaMemcpy(dev_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, h_b, ARRAY_BYTES, cudaMemcpyHostToDevice);
	
	cudaEventRecord(start, 0);
	vecAdd <<< 1, ARRAY_SIZE >>> (dev_a, dev_b, dev_c, ARRAY_SIZE);
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);

	cudaMemcpy(h_c, dev_c, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	for (i = 0; i < ARRAY_SIZE; i++)
	{
		printf("%f * %f = %f \n", h_a[i], h_b[i], h_c[i]);
	}
	printf("Time taken by the GPU is : %f\n", time);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}















