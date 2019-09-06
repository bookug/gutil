/*=============================================================================
# Filename: init.cu
# Author: bookug
# Mail: bookug@qq.com
# Last Modified: 2019-08-30 22:55
# Description: utilities of GPU
https://docs.nvidia.com/cuda/cusparse/index.html
    //NOTICE: the limit of one dimension blocks number is 65535
    //https://stackoverflow.com/questions/9841111/what-is-the-maximum-block-count-possible-in-cuda
//CUDA ERROR MESSAGE OF LAUNCHING KERNELS
https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038
//CUDA MEMCHECK
https://docs.nvidia.com/cuda/cuda-memcheck/index.html
=============================================================================*/

#include "init.h"

using namespace std; 


void 
gutil::initGPU(int dev)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    cudaSetDevice(dev);
	//NOTE: 48KB shared memory per block, 1024 threads per block, 30 SMs and 128 cores per SM
    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %luB; compute v%d.%d; clock: %d kHz; shared mem: %dB; block threads: %d; SM count: %d\n",
               devProps.name, devProps.totalGlobalMem, 
               (int)devProps.major, (int)devProps.minor, 
               (int)devProps.clockRate,
			   devProps.sharedMemPerBlock, devProps.maxThreadsPerBlock, devProps.multiProcessorCount);
    }
	cout<<"GPU selected"<<endl;
	//GPU initialization needs several seconds, so we do it first and only once
	//https://devtalk.nvidia.com/default/topic/392429/first-cudamalloc-takes-long-time-/
	int* warmup = NULL;
  //  unsigned long bigg = 0x7fffffff;
//    cudaMalloc(&warmup, bigg);
//    cout<<"warmup malloc"<<endl;
    //NOTICE: if we use nvprof to time the API calls, we will find the time of cudaMalloc() is very long.
    //The reason is that we do not add cudaDeviceSynchronize() here, so it is asynchronously and will include other instructions' time.
    //However, we do not need to add this synchronized function if we do not want to time the API calls
	cudaMalloc(&warmup, sizeof(int));
	cudaFree(warmup);
	cout<<"GPU warmup finished"<<endl;
    //heap corruption for 3 and 4
//    size_t size = 0x7fffffff;    //size_t is unsigned long in x64
    unsigned long size = 0x7fffffff;   //approximately 2G
 //   size *= 3;   
    size *= 4;
//    size *= 2;
	//NOTICE: the memory alloced by cudaMalloc is different from the GPU heap(for new/malloc in kernel functions)
//    cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
	cout<<"check heap limit: "<<size<<endl;

	// Runtime API
	// cudaFuncCachePreferShared: shared memory is 48 KB
	// cudaFuncCachePreferEqual: shared memory is 32 KB
	// cudaFuncCachePreferL1: shared memory is 16 KB
	// cudaFuncCachePreferNone: no preference
//    cudaFuncSetCacheConfig(MyKernel, cudaFuncCachePreferShared)
	//The initial configuration is 48 KB of shared memory and 16 KB of L1 cache
	//The maximum L2 cache size is 3 MB.
	//also 48 KB read-only cache: if accessed via texture/surface memory, also called texture cache;
	//or use _ldg() or const __restrict__
	//4KB constant memory, ? KB texture memory. cache size?
	//CPU的L1 cache是根据时间和空间局部性做出的优化，但是GPU的L1仅仅被设计成针对空间局部性而不包括时间局部性。频繁的获取L1不会导致某些数据驻留在cache中，只要下次用不到，直接删。
	//L1 cache line 128B, L2 cache line 32B, notice that load is cached while store not
	//mmeory read/write is in unit of a cache line
	//the word size of GPU is 32 bits
    //Titan XP uses little-endian byte order
}


