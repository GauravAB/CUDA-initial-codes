#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#define MAX 65535

#define imin(a,b) (a<b?a:b)

const int arr_size =8;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32,(arr_size +threadsPerBlock -1)/threadsPerBlock);

__global__ void kernel(float*arrA , float* arrB, float* arrC)
{
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float temp = 0;

	while (tid < arr_size)
	{
		temp += arrA[tid] * arrB[tid];
		tid += blockIdx.x * blockDim.x;
	}

	//set cache values 
	cache[cacheIndex] = temp;

	__syncthreads();

	//REDUCTION FUNCTION
	int i = blockDim.x / 2;

	while (i != 0)
	{
		if (cacheIndex < i)
		{
			cache[cacheIndex] += cache[cacheIndex + i];
			
		}
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
	{
		arrC[blockIdx.x] = cache[0];
	}
}


int main(int argc, char **argv)
{
	
	const int arr_bytes = arr_size * sizeof(float);


	float arr_a[MAX];
	float arr_b[MAX];
	float partial_c[MAX];

	float* dev_a;
	float* dev_b;
	float* partialdev_c;

	int i;
	float j = 1.0;

	for (i = 0; i < arr_size; i++)
	{
		arr_a[i] = j;
		arr_b[i] = j * j;
	}

	cudaMalloc((void**)&dev_a, arr_bytes);
	cudaMalloc((void**)&dev_b, arr_bytes);
	cudaMalloc((void**)&partialdev_c, blocksPerGrid * sizeof(float));

	cudaMemcpy(dev_a, arr_a, arr_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, arr_b, arr_bytes, cudaMemcpyHostToDevice);

	kernel <<<blocksPerGrid,threadsPerBlock >>>(dev_a,dev_b,partialdev_c);
	
	cudaMemcpy(partial_c, partialdev_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);


	//calculate final dot product on cpu side

    float c = 0;

	for (i = 0; i < blocksPerGrid; i++)
	{
		c += partial_c[i];
	}

	printf("The value of Dot product is : %f\n", c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(partialdev_c);

}





















