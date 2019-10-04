/*=============================================================================
# Filename: gutil.cuh
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

#ifndef _GUTIL_GUTIL_H
#define _GUTIL_GUTIL_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>

#include <cusparse.h>

#include <cub/cub.cuh> 

#include <iostream> 
#include "../util/Util.h" 

using namespace std; 




//DEBUG UTILITY

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}



//MEMORY UTILITY

void 
safeCudaMalloc(unsigned** addr, unsigned long bytes)
{
    if(bytes > 0)
    {
        cudaMalloc(addr, bytes);
    }
    else
    {
        *addr = NULL;
    }
}

unsigned* 
cudaCalloc(unsigned long bytes)
{
    unsigned* dv = NULL;
	safeCudaMalloc(&dv, bytes);
    cudaMemset(dv, 0, bytes);
    return dv;
}


unsigned*
newCopy(unsigned* src, unsigned size)
{
    unsigned* dst = NULL;
    safeCudaMalloc(&dst, sizeof(unsigned)*size);
    cudaMemcpy(dst, src, sizeof(unsigned)*size, cudaMemcpyDeviceToDevice);
    return dst;
}


void 
copyHtoD(unsigned*& d_ptr, unsigned* h_ptr, unsigned bytes)
{
    unsigned* p = NULL;
    safeCudaMalloc(&p, bytes);
    cudaMemcpy(p, h_ptr, bytes, cudaMemcpyHostToDevice);
    d_ptr = p;
    checkCudaErrors(cudaGetLastError());
}

void 
copyDtoH(unsigned*& h_ptr, unsigned* d_ptr, unsigned bytes)
{
    h_ptr = new unsigned[bytes];
    cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}



//SCAN UTILITY

unsigned
exclusive_sum(unsigned* d_array, unsigned size)
{
    // Determine temporary device storage requirements
    unsigned     *d_temp_storage = NULL; //must be set to distinguish two phase
    /*void* p = d_temp_storage;*/
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_array, d_array, size);
    // Allocate temporary storage
    safeCudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_array, d_array, size);
    cudaFree(d_temp_storage);
    unsigned ret;
    cudaMemcpy(&ret, d_array+size-1, sizeof(unsigned), cudaMemcpyDeviceToHost);
    return ret;
}

//NOTICE: this function only works for values of unsigned type.
//WARN: it must be called by the whole warp. (32)
__device__ unsigned
warpScan(unsigned presum, unsigned* exclusive_total = NULL)
{
    unsigned idx = threadIdx.x & 0x1f;  //thread index within the warp
    //perform inclusive prefix-sum scan
    for(unsigned stride = 1; stride < 32; stride <<= 1)
    {
        unsigned tmp = __shfl_up(presum, stride);
        if(idx >= stride)
        {
            presum += tmp;
        }
    }
    if(exclusive_total != NULL)
    {
        //perform exclusive prefix-sum scan
        *exclusive_total = __shfl(presum, 31);    //broadcast 
        presum = __shfl_up(presum, 1);
        if(idx == 0)
        {
            presum = 0;
        }
    }
    return presum;
}


//results are place in shared memory
//WARN: it must be called by the whole block. (1024)
__device__ unsigned
blockScan(volatile unsigned* s_cache, unsigned pred)
{
    s_cache[threadIdx.x] = pred;
    __syncthreads();
    //sequential prefix-sum for check
    /*if(threadIdx.x == 0)*/
    /*{*/
        /*unsigned base = 0;*/
        /*for(int i = 0; i < 1024; ++i)*/
        /*{*/
            /*unsigned tmp = s_cache[i];*/
            /*s_cache[i] = base;*/
            /*base += tmp;*/
        /*}*/
        /*s_cache[0] = base;*/
    /*}*/
    /*__syncthreads();*/
    /*unsigned ret = s_cache[0];*/
    /*__syncthreads();*/
    /*if(threadIdx.x == 0)*/
    /*{*/
        /*s_cache[0] = 0;*/
    /*}*/
    /*return ret;*/
    for(unsigned stride = 1; stride < 1024; stride <<= 1)
    {
        unsigned tmp = s_cache[threadIdx.x];
        if(threadIdx.x >= stride)
        {
            tmp = s_cache[threadIdx.x - stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride)
        {
            s_cache[threadIdx.x] += tmp;
        }
        __syncthreads();
    }
    unsigned total = s_cache[1023];
    unsigned tmp = 0;
    if(threadIdx.x > 0)
    {
        tmp = s_cache[threadIdx.x - 1];
    }
    __syncthreads();
    s_cache[threadIdx.x] = tmp;
    //NOTICE: below is not needed, if later a thread only accesses its own s_cache value.
    /*__syncthreads();*/
    return total;
}





//SEARCH UTILITY

__device__ unsigned
linear_search(unsigned _key, unsigned* _array, unsigned _array_num)
{
    for(int i = 0; i < _array_num; ++i)
    {
        if(_array[i] == _key)
        {
            return i;
        }
    }
    return INVALID;
}

//BETTER: maybe we can use dynamic parallism here
__device__ unsigned
binary_search(unsigned _key, unsigned* _array, unsigned _array_num)
{
//http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#host
/*#if defined(__CUDA_ARCH__)*/
	//MAYBE: return the right border, or use the left border and the right border as parameters
    if (_array_num == 0 || _array == NULL)
    {
		return INVALID;
    }

    unsigned _first = _array[0];
    unsigned _last = _array[_array_num - 1];

    if (_last == _key)
    {
        return _array_num - 1;
    }

    if (_last < _key || _first > _key)
    {
		return INVALID;
    }

    unsigned low = 0;
    unsigned high = _array_num - 1;
    unsigned mid;
    while (low <= high)
    {
        mid = (high - low) / 2 + low;   //same to (low+high)/2
        if (_array[mid] == _key)
        {
            return mid;
        }
        if (_array[mid] > _key)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
        }
    }
	return INVALID;
/*#else*/
/*#endif*/
}




//SET UTILITY

__global__ void
gather_kernel(unsigned* d_status, unsigned dsize, unsigned* d_cand)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= dsize)
	{
		return; 
	}
    int pos1 = d_status[i], pos2 = d_status[i+1];
    if(pos1 < pos2)
    {
        d_cand[pos1] = i;
    }
}

__global__ void
scatter_kernel(unsigned* d_status, unsigned* d_cand, unsigned dsize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= dsize)
	{
		return; 
	}
    int id = d_cand[i];
    d_status[id] = 1;
}

//int *result = new int[1000];
/*int *result_end = thrust::set_intersection(A1, A1 + size1, A2, A2 + size2, result, thrust::less<int>());*/
//
//BETTER: choose between merge-join and bianry-search, or using multiple threads to do intersection
//or do inetrsection per-element, use compact operation finally to remove invalid elements
__device__ void
intersect(unsigned*& cand, unsigned& cand_num, unsigned* list, unsigned list_num)
{
	int i, cnt = 0;
	for(i = 0; i < cand_num; ++i)
	{
        unsigned key = cand[i];
		unsigned found = binary_search(key, list, list_num);
		if(found != INVALID)
		{
			cand[cnt++] = key;
		}
	}
	cand_num = cnt;
}

__device__ void
subtract(unsigned*& cand, unsigned& cand_num, unsigned* record, unsigned result_col_num)
{
    //DEBUG: this will cause error when using dynamic allocation, gowalla with q0
	int i, j, cnt = 0;
    for(j = 0; j < cand_num; ++j)
    {
        unsigned key = cand[j];
        for(i = 0; i < result_col_num; ++i)
        {
            if(record[i] == key)
            {
                break;
            }
        }
        if(i == result_col_num)
        {
            cand[cnt++] = key;
        }
    }
	cand_num = cnt;
}



// GPU UTILITY

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};


//use watch -n 1 nvidia-smi instead of nvidia-smi -l 1
//nvidia-smi --format=csv --query-gpu=power.draw,utilization.gpu,fan.speed,temperature.gpu
//nvidia-smi --help-query-gpu
//https://github.com/wookayin/gpustat
//nvidia-smi  --format=csv --query-gpu=memory.used -i 0
//https://developer.nvidia.com/nvidia-system-management-interface
//BETTER: how to find memory usage of the corresponding program?
//cudaMemGetInfo()
//https://devtalk.nvidia.com/default/topic/1063443/cuda-programming-and-performance/different-cuda-memory-usage-between-nvidia-smi-and-cudamemgetinfo/
int
gpuMemUsage(int dev)
{
    string cmd = "nvidia-smi --format=csv --query-gpu=memory.used -i ";
    cmd = cmd + Util::int2string(dev) + " | tail -1 | awk '{print $1}'"; 
    cout<<"check: "<<cmd<<endl;
    string s = Util::getSystemOutput(cmd);
    int ret = Util::string2int(s);
    return ret;    //MB
}


//cuda map
//https://devtalk.nvidia.com/default/topic/523766/cuda-programming-and-performance/std-map-in-device-code/

//struct/class on gpu
//https://devtalk.nvidia.com/default/topic/1063212/cuda-programming-and-performance/performance-of-passing-structs-to-kernel-by-value-by-reference/

//cpu/gpu full performance
//https://elinux.org/Jetson/Performance

//gpu L2 cache
//https://devtalk.nvidia.com/default/topic/1063246/cuda-programming-and-performance/gpu-cache-coherence-problem/




#endif //_GUTIL_GUTIL_H




