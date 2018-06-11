#include <wb.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 512 
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float *aux, int len) {
	//@@ Modify the body of this kernel to generate the scanned blocks
	//@@ Make sure to use the workefficient version of the parallel scan
	//@@ Also make sure to store the block sum to the aux array 

	__shared__ float workspace[20 * BLOCK_SIZE];

	int p = 2 * blockIdx.x*blockDim.x + threadIdx.x;
	int q = p;
	int r = 2 * blockIdx.x*blockDim.x + blockDim.x - 1;

	//load
	for (int i = threadIdx.x; i < len; i += 2 * BLOCK_SIZE) {
		if (q < len)
			workspace[i] = input[q];
		if (q + blockDim.x < len)
			workspace[i + BLOCK_SIZE] = input[q + blockDim.x];
		q += 2 * BLOCK_SIZE;
	}

	__syncthreads();

	//reduce
	for (unsigned int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index <= 2 * BLOCK_SIZE)
			workspace[index] += workspace[index - stride];
	}

	//after reduction
	for (int stride = 2 * BLOCK_SIZE / 4; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < 2 * BLOCK_SIZE)
			workspace[index + stride] += workspace[index];
	}

	__syncthreads();

	if (p < len)
		output[p] = workspace[threadIdx.x];
	if (p + blockDim.x < len) {

		if (p + BLOCK_SIZE == r + BLOCK_SIZE) {
			output[p + BLOCK_SIZE] = workspace[threadIdx.x + BLOCK_SIZE] - workspace[threadIdx.x + BLOCK_SIZE];
			output[p + BLOCK_SIZE] = workspace[threadIdx.x + BLOCK_SIZE - 1] + input[p + BLOCK_SIZE];
			aux[blockIdx.x] = workspace[threadIdx.x + BLOCK_SIZE - 1] + input[p + BLOCK_SIZE];
		}
		else
			output[p + BLOCK_SIZE] = workspace[threadIdx.x + BLOCK_SIZE];
	}
}

__global__ void addScannedBlockSums(float *input, float *aux, int len) {
	//@@ Modify the body of this kernel to add scanned block sums to 
	//@@ all values of the scanned blocks

	int p = 2 * blockIdx.x * blockDim.x + threadIdx.x;

	if (blockIdx.x >= 1) {
		if (p < len)
			input[p] += aux[blockIdx.x - 1];
		if (p + blockDim.x < len)
			input[p + blockDim.x] += aux[blockIdx.x - 1];
	}

	__syncthreads();
}

int main(int argc, char **argv) {
	wbArg_t args;
	float *hostInput;  // The input 1D list
	float *hostOutput; // The output 1D list
	float *deviceInput;
	float *deviceOutput;
	float *deviceAuxArray, *deviceAuxScannedArray;
	int numElements, numBlocks;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
	hostOutput = (float *)malloc(numElements * sizeof(float));
	numBlocks = (float(numElements - 1)) / BLOCK_SIZE + 1;
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The number of input elements in the input is ", numElements);

	//@@ Allocate device memory
	//you can assume that aux array size would not need to be more than BLOCK_SIZE*2 (i.e., 1024)
	wbTime_start(GPU, "Allocating device memory.");
	cudaMalloc((void**)&deviceInput, numElements * sizeof(float));
	cudaMalloc((void**)&deviceOutput, numElements * sizeof(float));
	cudaMalloc((void**)&deviceAuxArray, 2 * numBlocks * sizeof(float));
	cudaMalloc((void **)&deviceAuxScannedArray, 2 * numBlocks * sizeof(float));
	wbTime_stop(GPU, "Allocating device memory.");

	wbTime_start(GPU, "Clearing output device memory.");
	wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
	wbTime_stop(GPU, "Clearing output device memory.");

	//@@ Copy input host memory to device	
	wbTime_start(GPU, "Copying input host memory to device.");
	cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(GPU, "Copying input host memory to device.");

	//@@ Initialize the grid and block dimensions here
	dim3 GridDim(numBlocks, 1, 1);
	dim3 BlockDim(BLOCK_SIZE, 1, 1);

	//@@ Modify this to complete the functionality of the scan
	//@@ on the deivce
	//@@ You need to launch scan kernel twice: 1) for generating scanned blocks 
	//@@ (hint: pass deviceAuxArray to the aux parameter)
	//@@ and 2) for generating scanned aux array that has the scanned block sums. 
	//@@ (hint: pass NULL to the aux parameter)
	//@@ Then you should call addScannedBlockSums kernel.
	wbTime_start(Compute, "Performing CUDA computation");
	scan <<< GridDim, BlockDim >>> (deviceInput, deviceOutput, deviceAuxArray, numElements);
	cudaDeviceSynchronize();
	scan <<< GridDim, BlockDim >>> (deviceAuxArray, deviceAuxScannedArray, NULL, numBlocks);
	cudaDeviceSynchronize();
	addScannedBlockSums <<< GridDim, BlockDim >>> (deviceOutput, deviceAuxScannedArray, numElements);
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	//@@ Copy results from device to host	
	wbTime_start(Copy, "Copying output device memory to host");
	cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying output device memory to host");

	//@@ Deallocate device memory
	wbTime_start(GPU, "Freeing device memory");
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	wbTime_stop(GPU, "Freeing device memory");

	wbSolution(args, hostOutput, numElements);

	free(hostInput);
	free(hostOutput);

	return 0;
}
