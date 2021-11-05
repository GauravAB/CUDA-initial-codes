#include "cuda_runtime.h";
#include "device_launch_parameters.h";
#include <stdio.h>


__global__ void cube_Func(float *in, float *out ,int N)
{
	int Idx = threadIdx.x;
	if (Idx < N)
	{
		float f = in[Idx];
		out[Idx] = f * f * f;
	}
}

void main(void)
{
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	float h_in[ARRAY_SIZE];
	float h_out[ARRAY_SIZE];
	float *dev_in;
	float *dev_out;
	int i;
	float f = 0.0;

	for (i = 0; i < ARRAY_SIZE; i++)
	{
		h_in[i] = f;
		f++;
	}

	cudaMalloc((void**)&dev_in, ARRAY_BYTES);
	cudaMalloc((void**)&dev_out, ARRAY_BYTES);
	
	cudaMemcpy(dev_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	cube_Func<<<1,ARRAY_SIZE>>>(dev_in, dev_out , ARRAY_SIZE);

	cudaMemcpy(h_out,dev_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	for (i = 0; i < ARRAY_SIZE; i++)
	{
		printf("cubed no. %d = %f\n",i,h_out[i]);
	}
}
