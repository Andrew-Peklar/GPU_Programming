#include <wb.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"

#define T_Width 16

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
  //@@ Insert code to implement tiled matrix multiplication here
  //@@ You have to use shared memory to write this kernel

	__shared__ float A_dim[T_Width][T_Width];
	__shared__ float B_dim[T_Width][T_Width];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = by * T_Width + ty;
	int Col = bx * T_Width + tx;

	float c_val = 0;

	for (int ch = 0; ch < (T_Width + numBRows - 1) / T_Width; ch++) {

		//for the A dimension
		if (Row < numARows && (ch * T_Width + tx < numAColumns))
			A_dim[ty][tx] = A[Row * numAColumns + ch * T_Width + tx];
		else
			A_dim[ty][tx] = 0.0;;


		//for the B dimension
		if (ch * T_Width + ty < numAColumns && Col < numBColumns)
			B_dim[ty][tx] = B[(ch * T_Width + ty) * numBColumns + Col];
		else
			B_dim[ty][tx] = 0.0;

		//synchronize
		__syncthreads();

		for (int j = 0; j < T_Width; j++)	c_val += A_dim[ty][j] * B_dim[j][tx];

		//synch again
		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns)
		C[((by*blockDim.y + ty)*numCColumns) + (bx*blockDim.x) + tx] = c_val;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  
  hostC = NULL;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  int Allo_C	= numCRows * numCColumns * sizeof(float);
  hostC			= (float*)malloc(numCRows * numCColumns * sizeof(float));

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);


  //@@ Allocate GPU memory here
  wbTime_start(GPU, "Allocating GPU memory.");

  int Allo_A = sizeof(float) * numARows * numAColumns;
  int Allo_B = sizeof(float) * numBRows * numBColumns;

  cudaMalloc((void **)&deviceA, Allo_A);
  cudaMalloc((void **)&deviceB, Allo_B);
  cudaMalloc((void **)&deviceC, Allo_C);

  wbTime_stop(GPU, "Allocating GPU memory.");


  //@@ Copy memory to the GPU here
  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceA, hostA, Allo_A, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, Allo_B, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(T_Width, T_Width, 1); //same initialization as BasicMatricMultiplication
  dim3 DimGrid((numBColumns - 1) / T_Width + 1, (numARows - 1) / T_Width + 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock >>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");


  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostC, deviceC, Allo_C, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");


  //@@ Free the GPU memory here
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
