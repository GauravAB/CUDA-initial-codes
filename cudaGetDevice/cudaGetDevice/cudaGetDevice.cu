#include <stdio.h>


int main(void)
{
	int dev;
	int no_dev;
	cudaDeviceProp prop;
	cudaGetDevice(&dev);
	cudaGetDeviceCount(&no_dev);
	cudaGetDeviceProperties(&prop, dev);
	printf("number of devices are %d\n", no_dev);
	printf("Id of current CUDA device %d\n", dev);
	printf("Number of maximum Threads per Block are: %d\n", prop.maxThreadsPerBlock);
	prop.major = 4;
	prop.minor = 0;

	cudaChooseDevice(&dev, &prop);
	printf("Id of chosen device id with closest match %d\n", dev);
	cudaSetDevice(dev);
}