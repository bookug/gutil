/*=============================================================================
# Filename: matrix.cuh
# Author: bookug
# Mail: bookug@qq.com
# Last Modified: 2019-08-30 22:55
# Description: implementation of matrix operations on GPU, with cuSPARSE library.
(Other choices include cuBLAS for dense matrix, cusp based on Thrust library.)
This is for graph algorithms, thus all matrixes are squares.
https://docs.nvidia.com/cuda/cusparse/index.html
    //NOTICE: the limit of one dimension blocks number is 65535
    //https://stackoverflow.com/questions/9841111/what-is-the-maximum-block-count-possible-in-cuda
//CUDA ERROR MESSAGE OF LAUNCHING KERNELS
https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038
//CUDA MEMCHECK
https://docs.nvidia.com/cuda/cuda-memcheck/index.html
=============================================================================*/

#ifndef _GUTIL_MATRIX_H
#define _GUTIL_MATRIX_H

#include "gutil.cuh"

using namespace std; 




//NOTICE: when COO structures are sorted by s first and t next, the indices array and values array are the same as CSR.
class Matrix    //square matrix
{
public:
    int n;  //the vertex num, dimension of x and y
    int m;  //the edge num, num of non-zeros
    unsigned* ro;  //row offset (or col offset)
    unsigned* ci; //col index (or row index)
    unsigned* ev;  //edge value
    Matrix()
    {
        n = m = 0;
        ro = ci = ev = NULL;
    }
    ~Matrix()
    {
        cudaFree(ro);
        cudaFree(ci);
        cudaFree(ev);
    }
    void copy(Matrix* m1)
    {
        cudaFree(ro);
        cudaFree(ci);
        cudaFree(ev);
        n = m1->n; m = m1->m;
        ro = newCopy(m1->ro, n+1);
        ci = newCopy(m1->ci, m);
        ev = newCopy(m1->ev, m);
    }
    void print()
    {
        unsigned *h_ro, *h_ci, *h_ev;
        copyDtoH(h_ro, ro, sizeof(unsigned)*(n+1));
        copyDtoH(h_ci, ci, sizeof(unsigned)*m);
        copyDtoH(h_ev, ev, sizeof(unsigned)*m);
        cout<<"print matrix"<<endl;
        cout<<"ro: ";
        for(int i = 0; i <= n; ++i)
        {
            cout<<h_ro[i]<<" ";
        }cout<<endl;
        cout<<"ci: ";
        for(int i = 0; i < m; ++i)
        {
            cout<<h_ci[i]<<" ";
        }cout<<endl;
        cout<<"ev: ";
        for(int i = 0; i < m; ++i)
        {
            cout<<h_ev[i]<<" ";
        }cout<<endl;
        delete[] h_ro;
        delete[] h_ci;
        delete[] h_ev;
    }
    void check(int v1, int v2)
    {
        cout<<"to check "<<v1<<" "<<v2<<endl;
        unsigned *h_ro, *h_ci, *h_ev;
        copyDtoH(h_ro, ro, sizeof(unsigned)*(n+1));
        copyDtoH(h_ci, ci, sizeof(unsigned)*m);
        copyDtoH(h_ev, ev, sizeof(unsigned)*m);
        int begin = h_ro[v1], end = h_ro[v1+1];
        for(int i = begin; i < end; ++i)
        {
            if(h_ci[i] == v2)
            {
                cout<<"find key-value "<<h_ev[i]<<endl;
                return;
            }
        }
        cout<<"not found key-value"<<endl;
    }
};



__global__ void
diag_kernel(unsigned* dv, int n, unsigned* ro, unsigned* ci, unsigned* ev)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n)
	{
		return; 
	}
    ro[i] = i;
    if(i == 0)
    {
        ro[n] = n;
    }
    ci[i] = i;
    ev[i] = dv[i];
}

__global__ void
any_kernel(unsigned* dv, int n, unsigned* ro)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n)
	{
		return; 
	}
    unsigned flag = 0;
    if(ro[i] != ro[i+1])
    {
        flag = 1;
    }
    dv[i] = flag;
}

__global__ void
spms_kernel(int m, unsigned* ev, int scalar)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= m)
	{
		return; 
	}
    ev[i] *= scalar;
}

__global__ void
ox_kernel(int m, unsigned* ev, int scalar)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= m)
	{
		return; 
	}
    if(ev[i] == scalar * scalar)
    {
        ev[i] = 1;
    }
    else
    {
        ev[i] = 0;
    }
}

__global__ void
spmm1_kernel(int n, unsigned* ro1, unsigned* ci1, unsigned* ev1, unsigned* ro2, unsigned* ci2, unsigned* ev2, unsigned* ro3)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n)
	{
		return; 
	}
    int v1 = i;
    int begin1 = ro1[v1], end1 = ro1[v1+1];
    for(int v2 = 0; v2 < n; ++v2)
    {
        int begin2 = ro2[v2], end2 = ro2[v2+1];
        unsigned sum = 0;
        /*assert(begin1 <= end1);*/
        /*assert(begin2 <= end2);*/
        /*printf("check %d\n", ci2[0]);*/
        for(int j = begin1; j < end1; ++j)
        {
            unsigned id = ci1[j];
            unsigned val = ev1[j];
            /*assert(ci2[0]>=0);*/
            /*unsigned ret = binary_search(id, ci2+begin2, end2-begin2);*/
            unsigned ret = linear_search(id, ci2+begin2, end2-begin2);
            if(ret != INVALID)
            {
                sum += val * ev2[ret+begin2];
            }
        }
        if(sum != 0)
        {
            ro3[v1]++;
        }
    }
}

__global__ void
spmm2_kernel(int n, unsigned* ro1, unsigned* ci1, unsigned* ev1, unsigned* ro2, unsigned* ci2, unsigned* ev2, unsigned* d_count, unsigned* ci3, unsigned* ev3)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n)
	{
		return; 
	}
    int v1 = i;
    int begin1 = ro1[v1], end1 = ro1[v1+1];
    for(int v2 = 0; v2 < n; ++v2)
    {
        int begin2 = ro2[v2], end2 = ro2[v2+1];
        unsigned sum = 0;
        /*if(v1 >= 26880)*/
        /*{*/
            /*printf("found large!\n");*/
        /*}*/
        /*if(v1 == 26880 && v2 == 27236)*/
        /*{*/
            /*printf("FOCUS!!!\n");*/
        /*}*/
        //BETTER: not reuse i for loop variable
        for(int j = begin1; j < end1; ++j)
        {
            unsigned id = ci1[j];
            unsigned val = ev1[j];
            /*unsigned ret = binary_search(id, ci2+begin2, end2-begin2);*/
            unsigned ret = linear_search(id, ci2+begin2, end2-begin2);
            if(ret != INVALID)
            {
                sum += val * ev2[ret+begin2];
            }
        /*if(v1 == 26880 && v2 == 27236 && id==27236)*/
        /*{*/
            /*printf("FOCUS: %d %d\n", ret, sum);*/
        /*}*/
        }
        if(sum != 0)
        {
            unsigned addr = d_count[v1];
            d_count[v1]++;
            ci3[addr] = v2;
            ev3[addr] = sum;
        }
    }
}

__global__ void
imatrix_kernel(unsigned* dv, int n)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n)
	{
		return; 
	}
    dv[i] = 1;
}




int
compress(unsigned* dv, int n, unsigned*& d_array)
{
    int ret;
    ret = exclusive_sum(dv, n+1); 
    unsigned* tmp;
    safeCudaMalloc(&tmp, sizeof(unsigned)*ret);
    d_array = tmp;
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
    gather_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(dv, n, d_array);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    return ret;
}

//one(i)
unsigned* 
one(int id, int n)
{
    unsigned* dv = (unsigned*)cudaCalloc(sizeof(unsigned)*n);
    cudaMemset(dv+id, 1, 1);  //works for little-endian machine
    /*int BLOCK_SIZE = 1024;*/
    /*int GRID_SIZE = (n+BLOCK_SIZE-1)/BLOCK_SIZE;*/
    /*one_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(id, n);*/
    /*cudaDeviceSynchronize();*/
    checkCudaErrors(cudaGetLastError());
    return dv;
}

//diag(vec)
void
diag(unsigned* dv, int n, Matrix* m1)
{
    m1->n = m1->m = n;
    safeCudaMalloc(&(m1->ro), sizeof(unsigned)*(n+1));
    safeCudaMalloc(&(m1->ci), sizeof(unsigned)*n);
    safeCudaMalloc(&(m1->ev), sizeof(unsigned)*n);
    //NOTICE: though dv may also contain 0s, but we do not consider it here.
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
    diag_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(dv, n, m1->ro, m1->ci, m1->ev);
    cudaDeviceSynchronize();
    //BETTER: we may reuse dv for ev
    checkCudaErrors(cudaGetLastError());
}


//M'  transpose    
void
transpose(Matrix* m1, Matrix* m2)
{
    /*cout<<"check transpose"<<endl;*/
    /*m1->print();*/
    m2->n = m1->n;
    m2->m = m1->m;
    int n = m1->n;
    int m = m1->m;
    safeCudaMalloc(&(m2->ro), sizeof(unsigned)*(n+1));
    safeCudaMalloc(&(m2->ci), sizeof(unsigned)*m);
    safeCudaMalloc(&(m2->ev), sizeof(unsigned)*m);
    //NOTICE: csr2csc, csc2csr
    cudaError_t cudaStat;
    cusparseStatus_t status;
    cusparseHandle_t handle=0;
    cusparseMatDescr_t descr=0;
    // initialize cusparse library 
    status= cusparseCreate(&handle);
    // create and setup matrix descriptor 
    status= cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    status = cusparseScsr2csc(handle, n, n, m, (float*)(m1->ev), (int*)(m1->ro), (int*)(m1->ci), (float*)(m2->ev), (int*)(m2->ci), (int*)(m2->ro), CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();
    if(status != CUSPARSE_STATUS_SUCCESS)
    {
        cout<<"error in transpose"<<endl;
    }
    status = cusparseDestroyMatDescr(descr);
    status = cusparseDestroy(handle);
    /*unsigned bufferSize = 0;*/
    /*status = cusparseCsr2cscEx2_bufferSize(handle, n, n, m, m1.ev, m1.ro, m1.ci, m2.ev, m2.ro, m2.ci, CUDA_R_32I, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &bufferSize);*/
    /*unsigned* buffer = NULL;*/
    /*safeCudaMalloc(&buffer, sizeof(unsigned)*bufferSize);*/
    /*status = cusparseCsr2cscEx2(handle, n, n, m, m1.ev, m1.ro, m1.ci, m2.ev, m2.ro, m2.ci, CUDA_R_32I, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer);*/
    /*cudaDeviceSynchronize();*/
    /*cudaFree(buffer);*/
    checkCudaErrors(cudaGetLastError());
    /*m2->print();*/
}

//any(Mat)
unsigned*
any(Matrix* m1)
{
    int n = m1->n;
    unsigned* ro = m1->ro;
    //n is the node num
    unsigned* dv = NULL;
    safeCudaMalloc(&dv, sizeof(unsigned)*(n+1));
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
    any_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(dv, n, ro);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    return dv;
}

void
diag_any(Matrix* m1, bool trans, Matrix* m2)
{
    unsigned* dv = NULL;
    Matrix tmpm;
    if(trans)
    {
        transpose(m1, &tmpm);
        dv = any(&tmpm);
    }
    else
    {
        dv = any(m1);
    }
    diag(dv, m1->n, m2);
    cudaFree(dv);
    checkCudaErrors(cudaGetLastError());
}

void
checkMM(Matrix* m1, Matrix* m2)
{
    /*int n = m1->n;*/
    /*unsigned* ro1 = new unsigned[n+1];*/
    /*unsigned* ci1 = new unsigned[m1->m];*/
    /*unsigned* ev1 = new unsigned[m1->m];*/
    /*unsigned* ro2 = new unsigned[n+1];*/
    /*unsigned* ci2 = new unsigned[m2->m];*/
    /*unsigned* ev2 = new unsigned[m2->m];*/
    /*cudaMemcpy(ro1, m1->ro, sizeof(unsigned)*(n+1), cudaMemcpyDeviceToHost);*/
    /*cudaMemcpy(ci1, m1->ci, sizeof(unsigned)*(m1->m), cudaMemcpyDeviceToHost);*/
    /*cudaMemcpy(ev1, m1->ev, sizeof(unsigned)*(m1->m), cudaMemcpyDeviceToHost);*/
    /*cudaMemcpy(ro2, m2.ro, sizeof(unsigned)*(n+1), cudaMemcpyDeviceToHost);*/
    /*cudaMemcpy(ci2, m2.ci, sizeof(unsigned)*(m2.m), cudaMemcpyDeviceToHost);*/
    /*cudaMemcpy(ev2, m2.ev, sizeof(unsigned)*(m2.m), cudaMemcpyDeviceToHost);*/
    /*for(int i = 0; i < n; ++i)*/
    /*{*/
        /*for(int j = 0; j < n; ++j)*/
        /*{*/
            /*int begin1 = ro1[i], end1 = ro1[i+1];*/
            /*int begin2 = ro2[i], end2 = ro2[i+1];*/
            /*int sum = 0;*/
            /*for(int k = begin1; k < end1; ++k)*/
            /*{*/
                /*int id = ci1[k];*/
                /*for(int s = begin2; s < end2; ++s)*/
                /*{*/
                    /*if(ci2[s] == id)*/
                    /*{*/
                        /*sum += ev1[k] * ev2[s];*/
                        /*if(i==0 && j == 0)*/
                        /*{*/
                            /*cout<<"check: "<<ev1[k]<<" "<<ev2[s]<<" "<<sum<<endl;*/
                        /*}*/
                        /*break;*/
                    /*}*/
                /*}*/
            /*}*/
            /*if(sum != 0)*/
            /*{*/
                /*cout<<"found non-zero "<<i<<" "<<j<<" "<<sum<<endl;*/
                /*return;*/
            /*}*/
        /*}*/
    /*}*/
    /*cout<<"non-zero is not found"<<endl;*/
    //free
}

//m3=m1xm2
void
spmm(Matrix* m1, Matrix* m2, Matrix* m3)
{
    /*cout<<"In spmm"<<endl;*/
    /*m1->print();*/
    /*m2->print();*/
    /*assert(m2.ci != NULL);*/
    if(m1->ci == NULL || m2->ci == NULL)
    {
        return;
    }
    int n = m1->n, m = m1->m;
    /*assert(n>0);*/
    Matrix m2c;
    transpose(m2, &m2c);
    //allocate space for m3
    m3->n = n;
    m3->ro = (unsigned*)cudaCalloc(sizeof(unsigned)*(n+1));
    checkCudaErrors(cudaGetLastError());
    int BLOCK_SIZE = 1024;
    //NOTICE: n*n may exceed the capacity of int, we should use long type for computing here.
    int GRID_SIZE = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
    spmm1_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, m1->ro, m1->ci, m1->ev, m2c.ro, m2c.ci, m2c.ev, m3->ro);
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    exclusive_sum(m3->ro, n+1);
    checkCudaErrors(cudaGetLastError());
    unsigned* d_count = newCopy(m3->ro, n);
    cudaMemcpy(&(m3->m), m3->ro+n, sizeof(unsigned), cudaMemcpyDeviceToHost);
    if(m3->m == 0)
    {
        cout<<"to check why zero in MM"<<endl;
        /*checkMM(m1, m2);*/
    }
    else
    {
        /*checkMM(m1, m2);*/
    }
    safeCudaMalloc(&(m3->ci), sizeof(unsigned)*(m3->m));
    safeCudaMalloc(&(m3->ev), sizeof(unsigned)*(m3->m));
    checkCudaErrors(cudaGetLastError());
    /*cout<<"grid size "<<GRID_SIZE<<endl;*/
    //WARN: too large block num, many blocks are dead.
    spmm2_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, m1->ro, m1->ci, m1->ev, m2c.ro, m2c.ci, m2c.ev, d_count, m3->ci, m3->ev);
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    cudaFree(d_count);

    /*cudaError_t cudaStat;*/
    /*cusparseStatus_t status;*/
    /*cusparseHandle_t handle=0;*/
    /*cusparseMatDescr_t descr=0;*/
    /*// initialize cusparse library */
    /*status= cusparseCreate(&handle);*/
    /*// create and setup matrix descriptor */
    /*status= cusparseCreateMatDescr(&descr);*/
    /*cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);*/
    /*cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);*/
    /*status= cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, n, m, 1, descr, m1.ev, m1.ro, m1.ci, denseBarray, n, 0, n, 0);*/
    /*status= cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, n, n, n, m, 1, descr, m1.ev, m1.ro, m1.ci, denseBarray, n, 0, n, 0);*/
    /*cudaDeviceSynchronize();*/
    /*//destroy matrix descriptor*/
    /*status = cusparseDestroyMatDescr(descr);*/
    /*//destroy handle*/
    /*status = cusparseDestroy(handle);*/
    checkCudaErrors(cudaGetLastError());
    /*m3->print();*/
}

//m1=m1xm2
void
spmm(Matrix* m1, Matrix* m2)
{
    Matrix m3;
    spmm(m1, m2, &m3);
    m1->copy(&m3);
}

//M OX S
void
spmm2(Matrix* m1, Matrix* m2, Matrix* m3, int scalar)
{
    spmm(m1, m2, m3);
    if(m3->ci == NULL)
    {
        return;
    }
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (m3->m+BLOCK_SIZE-1)/BLOCK_SIZE;
    ox_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(m3->m, m3->ev, scalar);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

//S * p
void 
spms(Matrix* m1, int scalar)
{
    if(m1->ci == NULL)
    {
        return;
    }
    int m = m1->m;
    unsigned* ev = m1->ev;
    //m is the edge num
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (m+BLOCK_SIZE-1)/BLOCK_SIZE;
    spms_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(m, ev, scalar);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

//S * p OX A
void 
spmsm2(Matrix* m1, Matrix* m2, Matrix* m3, int scalar)
{
    spms(m1, scalar);
    spmm2(m1, m2, m3, scalar);
    checkCudaErrors(cudaGetLastError());
}

void
newIMatrix(Matrix* m1, int n)
{
    m1->n = m1->m = n;
    unsigned* dv = NULL;
    safeCudaMalloc(&dv, sizeof(unsigned)*n);
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
    imatrix_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(dv, n);
    cudaDeviceSynchronize();
    diag(dv, n, m1);
    cudaFree(dv);
    checkCudaErrors(cudaGetLastError());
}

unsigned*
getIJ(Matrix* m1)
{
    unsigned* cooRowInd = NULL;
    int n = m1->n;
    int m = m1->m;
    safeCudaMalloc(&cooRowInd, sizeof(unsigned)*m);
    cudaError_t cudaStat;
    cusparseStatus_t status;
    cusparseHandle_t handle=0;
    cusparseMatDescr_t descr=0;
    status= cusparseCreate(&handle);
    status= cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    status = cusparseXcsr2coo(handle, (int*)(m1->ro), m, n,  (int*)cooRowInd, CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();
    if(status != CUSPARSE_STATUS_SUCCESS)
    {
        cout<<"error in csr2coo"<<endl;
    }
    status = cusparseDestroyMatDescr(descr);
    status = cusparseDestroy(handle);

    checkCudaErrors(cudaGetLastError());
    return cooRowInd;
}



//OPTIMIZATION OF DIAGONAL MATRIX MULTIPLICATION

__global__ void
selcol1_kernel(unsigned* d_addr, int size, unsigned* ro1, unsigned* ev1, unsigned* ro2, int p = -1)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= size)
	{
		return; 
	}
    int id = d_addr[i];
    int begin = ro1[id], end = ro1[id+1];
    if(p == -1)
    {
        ro2[id] = end - begin;
        return;
    }
    int cnt = 0;
    for(int j = begin; j < end; ++j)
    {
        if(ev1[j] == p)
        {
            cnt++;
        }
    }
    ro2[id] = cnt;
}

__global__ void
selcol2_kernel(unsigned* d_addr, int size, unsigned* ro1, unsigned* ci1, unsigned* ev1, unsigned* ro2, unsigned* ci2, unsigned* ev2, int p = -1)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= size)
	{
		return; 
	}
    int id = d_addr[i];
    int begin = ro1[id], end = ro1[id+1];
    int pos = ro2[id];
    for(int j = begin; j < end; ++j)
    {
        if(p == -1)
        {
            ci2[pos] = ci1[j];
            ev2[pos] = ev1[j];
            pos++;
        }
        else if(ev1[j] == p)
        {
            ci2[pos] = ci1[j];
            ev2[pos] = 1;
            pos++;
        }
    }
}


//A x DiagMatrix
//NOTICE: we let m1 be CSC format (the transpose of CSR format) and m2 be CSR format
void
column_select(Matrix* m1, unsigned* addr, unsigned size, Matrix* m2)
{
    int n = m1->n; 
    Matrix tmpm;
    tmpm.n = n;
    tmpm.ro = cudaCalloc(sizeof(unsigned)*(n+1));
    unsigned* d_addr = NULL;
    copyHtoD(d_addr, addr, sizeof(unsigned)*size);
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
    selcol1_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_addr, size, m1->ro, m1->ev, tmpm.ro);
    cudaDeviceSynchronize();
    tmpm.m = exclusive_sum(tmpm.ro, n+1);
    /*cudaMemcpy(&(tmpm.m), tmpm.ro+n, sizeof(unsigned), cudaMemcpyDeviceToHost);*/
    safeCudaMalloc(&(tmpm.ci), sizeof(unsigned)*tmpm.m);
    safeCudaMalloc(&(tmpm.ev), sizeof(unsigned)*tmpm.m);
    selcol2_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_addr, size, m1->ro, m1->ci, m1->ev, tmpm.ro, tmpm.ci, tmpm.ev);
    cudaDeviceSynchronize();
    cudaFree(d_addr);
    transpose(&tmpm, m2);
    checkCudaErrors(cudaGetLastError());
}

//DiagMatrix x A, an optional restriction is to extract only label-p edges.
//NOTICE: we let m1 and m2 be CSR format.
void
row_select(Matrix* m1, unsigned* addr, unsigned size, Matrix* m2, int p = -1)
{
    int n = m1->n; 
    m2->n = n;
    m2->ro = cudaCalloc(sizeof(unsigned)*(n+1));
    unsigned* d_addr = NULL;
    copyHtoD(d_addr, addr, sizeof(unsigned)*size);
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
    selcol1_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_addr, size, m1->ro, m1->ev, m2->ro, p);
    cudaDeviceSynchronize();
    m2->m = exclusive_sum(m2->ro, n+1);
    /*cudaMemcpy(&(m2->m), m2->ro+n, sizeof(unsigned), cudaMemcpyDeviceToHost);*/
    safeCudaMalloc(&(m2->ci), sizeof(unsigned)*m2->m);
    safeCudaMalloc(&(m2->ev), sizeof(unsigned)*m2->m);
    selcol2_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_addr, size, m1->ro, m1->ci, m1->ev, m2->ro, m2->ci, m2->ev, p);
    cudaDeviceSynchronize();
    cudaFree(d_addr);
    checkCudaErrors(cudaGetLastError());
}


#endif //_GUTIL_MATRIX_H




