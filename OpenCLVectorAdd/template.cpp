#include <wb.h>
#include <CL/opencl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//@@ Write the OpenCL kernel
const char *kernelSource = "\n"
"__kernel void vadd(__global float *d_A, __global float *d_B, __global float *d_C, int n){	\n"
"	int id = get_global_id(0);																\n"
"																							\n"
"	if(id < n){																				\n"
"		d_C[id] = d_A[id] + d_B[id];														\n"
"	}																						\n"
"																							\n"
"}																							\n";


int main(int argc, char *argv[]) {

	
  wbArg_t args;
  int inputLength;
  int inputLengthBytes;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
 
  //grid and block sizes
  size_t gridSize;
  size_t blockSize;

  //CL types
  cl_mem deviceInput1;
  cl_mem deviceInput2;
  cl_mem deviceOutput;
  cl_int succ = CL_SUCCESS; //<------- added
  
  deviceInput1 = NULL;
  deviceInput2 = NULL;
  deviceOutput = NULL;

  cl_platform_id cpPlatform; // OpenCL platform
  cl_device_id device_id;    // device ID
  cl_context context;        // context
  cl_command_queue queue;    // command queue
  cl_program program;        // program
  cl_kernel kernel;          // kernel

  context	= NULL;        
  queue		= NULL;    
  program	= NULL;
  kernel	= NULL;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1		= (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2		= (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  inputLengthBytes	= inputLength * sizeof(float);
  hostOutput		= (float *)malloc(inputLengthBytes);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The input size is ", inputLengthBytes, " bytes");

  //@@ Initialize the workgroup dimensions
  blockSize	= 512;
  gridSize = ceil(inputLength / (float)blockSize)*blockSize;

  //@@ Bind to platform
  succ = clGetPlatformIDs(1, &cpPlatform, NULL);

  //@@ Get ID for the device
  succ = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

  //@@ Create a context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &succ);

  //@@ Create a command queue
  queue = clCreateCommandQueue(context, device_id, 0, &succ);

  //@@ Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &succ);

  //@@ Build the program executable
  succ = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  //@@ Create the compute kernel in the program we wish to run
  kernel = clCreateKernel(program, "vadd", &succ);

  //@@ Create the input and output arrays in device memory for our calculation
  deviceInput1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLengthBytes, hostInput1, NULL);
  deviceInput2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLengthBytes, hostInput2, NULL);
  deviceOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, inputLengthBytes, NULL, NULL);;

  //@@ Write our data set into the input array in device memory
  succ = clEnqueueWriteBuffer(queue, deviceInput1, CL_TRUE, 0, inputLengthBytes, hostInput1, 0, NULL, NULL);
  succ = clEnqueueWriteBuffer(queue, deviceInput2, CL_TRUE, 0, inputLengthBytes, hostInput2, 0, NULL, NULL);

  //@@ Set the arguments to our compute kernel
  succ = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&deviceInput1);
  succ = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&deviceInput2);
  succ = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&deviceOutput);
  succ = clSetKernelArg(kernel, 3, sizeof(int), &inputLength);

  //@@ Execute the kernel over the entire range of the data set
  succ = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gridSize, &blockSize, 0, NULL, NULL);

  //@@ Wait for the command queue to get serviced before reading back results
  clFinish(queue);

  //@@ Read the results from the device
  clEnqueueReadBuffer(queue, deviceOutput, CL_TRUE, 0, inputLengthBytes, hostOutput, 0, NULL, NULL);

  wbSolution(args, hostOutput, inputLength);

  // release OpenCL resources
  clReleaseMemObject(deviceInput1);
  clReleaseMemObject(deviceInput2);
  clReleaseMemObject(deviceOutput);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  // release host memory
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
