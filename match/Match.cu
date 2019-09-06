/*=============================================================================
# Filename: Match.cpp
# Author: bookug
# Mail: bookug@qq.com
# Last Modified: 2016-12-15 01:38
# Description:  implementation of MAGiQ
# We use cuSPARSE library of CUDA.
https://docs.nvidia.com/cuda/cusparse/index.html
=============================================================================*/

#include "Match.h"

//NOTICE: below must be included here instead of the header. 
//The reason is that Match.h is included by main program, thus it can not contain CUDA syntaxes.
#include <cub/cub.cuh> 

#include "../gutil/matrix.cuh"

using namespace std;



Match::Match(Graph* _query, Graph* _data)
{
	this->query = _query;
	this->data = _data;
	id2pos = pos2id = NULL;
}

Match::~Match()
{
	delete[] this->id2pos;
}

inline void 
Match::add_mapping(int _id)
{
	pos2id[current_pos] = _id;
	id2pos[_id] = current_pos;
	this->current_pos++;
}

__host__ unsigned
binary_search_cpu(unsigned _key, unsigned* _array, unsigned _array_num)
{
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
}



int
Match::filter(int* qnum, int* cand)
{
    int qsize = this->query->vertex_num, dsize = this->data->vertex_num;
    int idx = -1;
    int minv = 2147483647;
    for(int i = 0; i < qsize; ++i)
    {
        int vlb = this->query->vertices[i].label;
        unsigned ret = binary_search_cpu(vlb, this->data->inverse_label, this->data->label_num);
            int num = this->data->inverse_offset[ret+1] - this->data->inverse_offset[ret];
            if(num < minv)
            {
                minv = num;
                idx = i;
            }
        qnum[i] = num;
        cand[i] = this->data->inverse_offset[ret];
    }
	return idx;
}




bool
dfs(Graph* q, int source, vector<DFSEdge>& ops, bool* visited)
{
    vector<Neighbor>& in = q->vertices[source].in;
    vector<Neighbor>& out = q->vertices[source].out;
    bool found = false;
    for(int i = 0; i < in.size(); ++i)
    {
        int eid = in[i].eid;
        if(visited[eid])
        {
            continue;
        }
        found = true;
        int vid = in[i].vid, elb = in[i].elb;
        ops.push_back(DFSEdge(eid, source, vid, elb, 0, 1));
        visited[eid] = true;
        bool ret = dfs(q, in[i].vid, ops, visited);
        if(ret)
        {
            ops.push_back(DFSEdge(eid, source, vid, elb, 1, 1));
        }
    }
    for(int i = 0; i < out.size(); ++i)
    {
        int eid = out[i].eid;
        if(visited[eid])
        {
            continue;
        }
        found = true;
        int vid = out[i].vid, elb = out[i].elb;
        ops.push_back(DFSEdge(eid, source, vid, elb, 0, 0));
        visited[eid] = true;
        bool ret = dfs(q, out[i].vid, ops, visited);
        if(ret)
        {
            ops.push_back(DFSEdge(eid, source, vid, elb, 1, 0));
        }
    }
    return found;
}

void
prepare_candidate(unsigned* addr, unsigned size, Matrix* m1, int n)
{
    unsigned* dv =  (unsigned*)cudaCalloc(sizeof(unsigned)*n);
    unsigned* d_addr = NULL;
    copyHtoD(d_addr, addr, sizeof(unsigned)*size);
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
    scatter_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(dv, d_addr, size);
    cudaDeviceSynchronize();
    cudaFree(d_addr);
    diag(dv, n, m1);
    cudaFree(dv);
    checkCudaErrors(cudaGetLastError());
}

void
acquire_candidate(Matrix* m1, bool trans, unsigned*& tmpv, unsigned& tmpn)
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
    unsigned* d_array = NULL;
    tmpn = compress(dv, m1->n, d_array);
    copyDtoH(tmpv, d_array, sizeof(unsigned)*tmpn);
    cudaFree(dv);
    cudaFree(d_array);
    checkCudaErrors(cudaGetLastError());
}

__global__ void
combine_kernel(unsigned* keys, unsigned* vals, unsigned m, unsigned* d_array)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= m)
	{
		return; 
	}
    int a = keys[i], b = vals[i];
    d_array[2*i] = a;
    d_array[2*i+1] = b;
}


void
combine(unsigned* keys, unsigned* vals, int m, unsigned*& d_array)
{
    unsigned* tmp;
    cudaMalloc(&tmp, sizeof(unsigned)*2*m);
    d_array = tmp;
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (m+BLOCK_SIZE-1)/BLOCK_SIZE;
    combine_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(keys, vals, m, d_array);
    cudaDeviceSynchronize();
}

__global__ void
join_kernel(unsigned* d_candidate, unsigned* d_result, unsigned* d_count, unsigned result_row_num, unsigned result_col_num, unsigned upos, unsigned vpos, unsigned array_num)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int group = t >> 5;
	int idx = t & 0x1f;
	if(group >= result_row_num)
	{
		return; 
	}
	
	unsigned mu = d_result[group*result_col_num+upos];
	unsigned mv = d_result[group*result_col_num+vpos];
	//find mu in d_array using a warp
	unsigned size = array_num;
	unsigned loop = size >> 5;
	size = size & 0x1f;

	unsigned flag = 0;
	unsigned base = 0;
	for(int j = 0; j < loop; ++j, base+=32)
	{
		unsigned v1 = d_candidate[2*(idx+base)];
		unsigned v2 = d_candidate[2*(idx+base)+1];
		if(v1 == mu && v2 == mv)
		{
			flag = 1;
		}
        flag = __any(flag);
		if(flag == 1)
		{
			break;
		}
	}
	if(flag == 0 && idx < size)
	{
		unsigned v1 = d_candidate[2*(idx+base)];
		unsigned v2 = d_candidate[2*(idx+base)+1];
		if(v1 == mu && v2 == mv)
		{
			flag = 1;
		}
	}
    flag = __any(flag);
	if(idx == 0)
	{
		d_count[group] = flag;
	}
}

__global__ void
filter_kernel(unsigned* d_result, unsigned* d_result_new, unsigned* d_count, unsigned result_row_num, unsigned result_col_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= result_row_num)
	{
		return; 
	}

	if(d_count[idx] < d_count[idx+1])  //this is a valid result
	{
		//write d_result[idx] to d_result_new[d_count[idx]]
		memcpy(d_result_new+d_count[idx]*result_col_num, d_result+idx*result_col_num, sizeof(unsigned)*result_col_num);
	}
}

void 
kernel_join(unsigned* d_candidate, unsigned array_num, unsigned*& d_result, unsigned& result_row_num, unsigned& result_col_num, unsigned upos, unsigned vpos)
{
	//follow the two-step output scheme to write the merged results
	unsigned* d_count = NULL;
	checkCudaErrors(cudaMalloc(&d_count, sizeof(unsigned)*(result_row_num+1)));
	/*join_kernel<<<(result_row_num*32+1023)/1024,1024>>>(d_candidate, d_result, d_count, result_row_num, result_col_num, upos, vpos, array_num);*/
    join_kernel<<<(result_row_num*32+1023)/1024,1024>>>(d_candidate, d_result, d_count, result_row_num, result_col_num, upos, vpos, array_num);
	cudaDeviceSynchronize();
	
	//prefix sum to find position
    unsigned sum = exclusive_sum(d_count, result_row_num+1);
	/*thrust::device_ptr<unsigned> dev_ptr(d_count);*/
	/*unsigned sum;*/
	/*thrust::exclusive_scan(dev_ptr, dev_ptr+result_row_num+1, dev_ptr);*/
	/*checkCudaErrors(cudaGetLastError());*/
	/*cudaMemcpy(&sum, d_count+result_row_num, sizeof(unsigned), cudaMemcpyDeviceToHost);*/
	if(sum == 0)
	{
		checkCudaErrors(cudaFree(d_count));
		checkCudaErrors(cudaFree(d_result));
		d_result = NULL;
		result_row_num = 0;
		return;
	}

	unsigned* d_result_new = NULL;
	checkCudaErrors(cudaMalloc(&d_result_new, sizeof(unsigned)*sum*result_col_num));
	//just one thread for each row is ok
	filter_kernel<<<(result_row_num+1023)/1024,1024>>>(d_result, d_result_new, d_count, result_row_num, result_col_num);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaFree(d_count));

	checkCudaErrors(cudaFree(d_result));
	d_result = d_result_new;
	//NOTICE: result_col_num not changes in this case
	result_row_num = sum;
}

__global__ void
expand_kernel(unsigned* d_candidate, unsigned* d_result, unsigned* d_count, unsigned result_row_num, unsigned result_col_num, unsigned pos, unsigned array_num)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	/*int group = t / 32;*/
	/*int idx = t % 32;*/
	int group = t >> 5;
    /*printf("group: %d\n", group);*/
	int idx = t & 0x1f;
	if(group >= result_row_num)
	{
		return; 
	}
	
	unsigned mu = d_result[group*result_col_num+pos];
	//find mu in d_array using a warp
	unsigned size = array_num;
	/*unsigned loop = size / 32;*/
	/*size = size % 32;*/
	unsigned loop = size >> 5;
	size = size & 0x1f;

	unsigned cnt = 0;
	unsigned base = 0;
	for(int j = 0; j < loop; ++j, base+=32)
	{
		unsigned v1 = d_candidate[2*(idx+base)];
		unsigned v2 = d_candidate[2*(idx+base)+1];
		if(v1 == mu)
		{
            cnt++;
		}
	}
	if(idx < size)
	{
		unsigned v1 = d_candidate[2*(idx+base)];
		unsigned v2 = d_candidate[2*(idx+base)+1];
		if(v1 == mu)
		{
            cnt++;
		}
	}
    //warp reduce sum
    unsigned total;
    cnt = warpScan(cnt, &total);
    if(idx == 0)
    {
        d_count[group] = total;
    }
}

__global__ void
link_kernel(unsigned* d_result, unsigned* d_result_new, unsigned* d_count, unsigned result_row_num, unsigned result_col_num, unsigned* d_candidate, unsigned pos, unsigned array_num)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int group = t >> 5;
	int idx = t & 0x1f;
	if(group >= result_row_num)
	{
		return; 
	}
    unsigned row_base = d_count[group];
	if(row_base == d_count[group+1])  //this is a invalid result
	{
		return; 
	}
	
	unsigned mu = d_result[group*result_col_num+pos];
	//find mu in d_array using a warp
	unsigned size = array_num;
	/*unsigned loop = size / 32;*/
	/*size = size % 32;*/
	unsigned loop = size >> 5;
	size = size & 0x1f;

	unsigned base = 0;
	unsigned write_base = row_base * (result_col_num+1);
    unsigned pred, presum, total;
    int v1, v2;
	for(int j = 0; j < loop; ++j, base+=32)
	{
		v1 = d_candidate[2*(idx+base)];
		v2 = d_candidate[2*(idx+base)+1];
        pred = 0;
		if(v1 == mu)
		{
            pred = 1;
		}
        presum = warpScan(pred, &total);
        if(pred == 1)
        {
            memcpy(d_result_new+write_base+presum*(result_col_num+1), d_result+group*result_col_num, sizeof(unsigned)*result_col_num);
            d_result_new[write_base+presum*(result_col_num+1)+result_col_num] = v2;
            write_base += total*(result_col_num+1);
        }
	}
    pred = 0;   //must be reset before judgement
	if(idx < size)
	{
		v1 = d_candidate[2*(idx+base)];
		v2 = d_candidate[2*(idx+base)+1];
		if(v1 == mu)
		{
            pred = 1;
		}
	}
    presum = warpScan(pred, &total); //must be called by the whole warp
    if(pred == 1)
    {
        memcpy(d_result_new+write_base+presum*(result_col_num+1), d_result+group*result_col_num, sizeof(unsigned)*result_col_num);
        d_result_new[write_base+presum*(result_col_num+1)+result_col_num] = v2;
        write_base += total*(result_col_num+1);
    }
    //we guarantee that mu and some mv must be found here, because the case that no result exists has been judged at the beginning
}

void 
kernel_expand(unsigned* d_candidate, unsigned array_num, unsigned*& d_result, unsigned& result_row_num, unsigned& result_col_num, unsigned pos)
{
	//follow the two-step output scheme to write the merged results
	unsigned* d_count = NULL;
	checkCudaErrors(cudaMalloc(&d_count, sizeof(unsigned)*(result_row_num+1)));
	expand_kernel<<<(result_row_num*32+1023)/1024,1024>>>(d_candidate, d_result, d_count, result_row_num, result_col_num, pos, array_num);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	//prefix sum to find position
    unsigned sum = exclusive_sum(d_count, result_row_num+1);
	/*thrust::device_ptr<unsigned> dev_ptr(d_count);*/
	/*unsigned sum;*/
	/*thrust::exclusive_scan(dev_ptr, dev_ptr+result_row_num+1, dev_ptr);*/
	/*checkCudaErrors(cudaGetLastError());*/
	/*cudaMemcpy(&sum, d_count+result_row_num, sizeof(unsigned), cudaMemcpyDeviceToHost);*/
	if(sum == 0)
	{
		checkCudaErrors(cudaFree(d_count));
		checkCudaErrors(cudaFree(d_result));
		d_result = NULL;
		result_row_num = 0;
		return;
	}

	unsigned* d_result_new = NULL;
	checkCudaErrors(cudaMalloc(&d_result_new, sizeof(unsigned)*sum*(result_col_num+1)));
	/*link_kernel<<<(result_row_num*32+1023)/1024,1024>>>(d_result, d_result_new, d_count, result_row_num, result_col_num, d_candidate, pos, array_num);*/
	link_kernel<<<(result_row_num*32+1023)/1024,1024>>>(d_result, d_result_new, d_count, result_row_num, result_col_num, d_candidate, pos, array_num);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_count));

	checkCudaErrors(cudaFree(d_result));
	d_result = d_result_new;
	result_row_num = sum;
	result_col_num++;
}

void 
Match::match(IO& io, unsigned*& final_result, unsigned& result_row_num, unsigned& result_col_num, int*& id_map)
{
	long t0 = Util::get_cur_time();

	int qsize = this->query->vertex_num;
    int n = this->data->vertex_num;
    int m = this->data->edge_num;
    assert(qsize <= 12);
    int qesize = this->query->edge_num;
    Matrix A, At;
    A.n = n;  A.m = m;
    //build on device : A (out csr) and A' (in csr)
    copyHtoD(A.ro, this->data->ro, sizeof(unsigned)*(n+1));
    copyHtoD(A.ci, this->data->ci, sizeof(unsigned)*m);
    copyHtoD(A.ev, this->data->ev, sizeof(unsigned)*m);
    transpose(&A, &At);
	long t1 = Util::get_cur_time();

    //translate the query
    vector<DFSEdge> ops; //positive for forward edges; negative for backward edges
    vector<Edge>& es = this->query->edges;
    int* qnum = new int[qsize+1];
    int* cand = new int[qsize+1];
	int source = filter(qnum, cand);
    bool* visited = new bool[qesize];
    memset(visited, 0, sizeof(bool)*(qesize));
    dfs(this->query, source, ops, visited);
    delete[] visited;

    bool success = true;
    vector<Matrix> ms(qesize);
    //filter candidate edges according to the operation sequence
    for(int i = 0; i < ops.size(); ++i)
    {
        /*cout<<"check matrix A"<<endl;*/
        /*A.print();*/
        /*cout<<"check matrix At"<<endl;*/
        /*At.print();*/
        int backward = ops[i].backward;
        int v1 = ops[i].v1, v2 = ops[i].v2, p = ops[i].p, eid = ops[i].eid;
        /*cout<<"the "<<i<<"-th operation: "<<v1<<" "<<v2<<" "<<p<<" "<<eid<<endl;*/
        Matrix S;
        Matrix in2;
        //WARN: we can not change the C++ reference, if using mp=At later, A will be overwritten by At.
        /*Matrix& mp = A;*/
        /*Matrix* mp = &A;*/
        Matrix* mp = &At;   //use CSC format here
        if(ops[i].reversed == 1)
        {
            /*mp = &At;*/
            mp = &A;
        }
        if(backward == 0)
        {
            Matrix tmpm;
            /*prepare_candidate(this->data->inverse_vertex+cand[v2], qnum[v2], &tmpm, n);*/
            /*spmm(mp, &tmpm, &in2);*/
            column_select(mp, this->data->inverse_vertex+cand[v2], qnum[v2], &in2);
#ifdef DEBUG
            mp->print();
            in2.print();
#endif
            /*if(i == 0)*/
            /*{*/
                /*mp->check(26880,27236);*/
                /*tmpm.check(27236,27236);*/
                /*in2.check(26880, 27236);*/
            /*}*/
            /*assert(in2.m != 0);*/
        }
        if(i == 0)  //the first edge
        {
            /*prepare_candidate(this->data->inverse_vertex+cand[v1], qnum[v1], &S, n);*/
            //Consider vertex label, we use diag(vec(C(l))) instead
            //newIMatrix(S, n);
            /*spmsm2(&S, &in2, &ms[eid], p);*/
            row_select(&in2, this->data->inverse_vertex+cand[v1], qnum[v1], &ms[eid], p);
#ifdef DEBUG
            in2.print();
            ms[eid].print();
#endif
            /*S.check(26880,26880);*/
            /*in2.check(26880,27236);*/
            /*ms[eid].check(26880,27236);*/
        }
        else if(backward == 0)
        {
            int peid = ops[i-1].eid;
            bool trans = (ops[i-1].v1 != v1);
            unsigned *tmpv, tmpn;
            acquire_candidate(&ms[peid], trans, tmpv, tmpn);
            row_select(&in2, tmpv, tmpn, &ms[eid], p);
#ifdef DEBUG
            ms[peid].print();
            in2.print();
            ms[eid].print();
#endif
            /*diag_any(&ms[peid], trans, &S);*/
            /*spmsm2(&S, &in2, &ms[eid], p);*/
        }
        else if(backward == 1)
        {
            int peid = ops[i-1].eid;
            unsigned *tmpv, tmpn;
            acquire_candidate(&ms[peid], false, tmpv, tmpn);
            Matrix tmpm; transpose(&ms[eid], &tmpm);
            column_select(&tmpm, tmpv, tmpn, &ms[eid]);
#ifdef DEBUG
            ms[peid].print();
            tmpm.print();
            ms[eid].print();
#endif
            /*diag_any(&ms[peid], false, &S);*/
            /*spmm(&ms[eid], &S);*/
        }
        if(ms[eid].m == 0)
        {
        cout<<"filter ends: "<<i<<" "<<v1<<" "<<v2<<" "<<p<<endl;
            success = false;
            break;
        }
    }

	long t2 = Util::get_cur_time();
	cout<<"filter used: "<<(t2-t1)<<"ms"<<endl;
    if(!success)
    {
        cout<<"filter failed!"<<endl;
        delete[] qnum;
        delete[] cand;
        release();
		final_result = NULL;
		result_row_num = 0;
		result_col_num = qsize;
        return;
    }
    //NOTICE: we ensure that the query has results.



    /*checkCudaErrors(cudaGetLastError());*/
    /*cudaFree(NULL);*/
    checkCudaErrors(cudaGetLastError());
	//initialize the mapping structure
	this->id2pos = new int[qsize];
	this->pos2id = new int[qsize];
	this->current_pos = 0;
	memset(id2pos, -1, sizeof(int)*qsize);
	memset(pos2id, -1, sizeof(int)*qsize);
	//intermediate table of join results
    unsigned* d_result = NULL;

    //Result Generation    
    //NOTICE: we use the order of forward edges in ops as the join order, thus it is always connected and no need to prepare reversed candidate edges.
    for(int i = 0; i < ops.size(); ++i)
    {
        if(ops[i].backward == 1)
        {
            continue;
        }
        int v1 = ops[i].v1, v2 = ops[i].v2, p = ops[i].p, eid = ops[i].eid;
        //two column table: cooRowInd is one column, ci is another column, both size-m vectors.
        unsigned* keys = getIJ(&ms[eid]);
        unsigned* vals = ms[eid].ci;
        unsigned m = ms[eid].m;
        if(this->current_pos == 0)  //the first edge candidate set
        {
            result_row_num = m;
            result_col_num = 2;
            combine(keys, vals, m, d_result);
            this->add_mapping(v1);
            this->add_mapping(v2);
            continue;
        }

        unsigned* d_candidate = NULL;
        combine(keys, vals, m, d_candidate);
        //relational join like GpSM
        bool mode = 0;   //expand mode
        int v1pos = this->id2pos[v1];
        assert(v1pos != -1);
        int v2pos = this->id2pos[v2];
        if(v2pos != -1)
        {
            mode = 1;  //join mode
        }
		if(mode == 0)
		{
				kernel_expand(d_candidate, m, d_result, result_row_num, result_col_num, v1pos);
				this->add_mapping(v2);
		}
		else
		{
			kernel_join(d_candidate, m, d_result, result_row_num, result_col_num, v1pos, v2pos);
		}

        cudaFree(keys);
		if(result_row_num == 0)
		{
            success = false;
            break;
		}
    }

	long t3 = Util::get_cur_time();
	//transfer the result to CPU and output
	if(success)
	{
		final_result = new unsigned[result_row_num * result_col_num];
		cudaMemcpy(final_result, d_result, sizeof(unsigned)*result_col_num*result_row_num, cudaMemcpyDeviceToHost);
	}
	else
	{
		final_result = NULL;
		result_row_num = 0;
		result_col_num = qsize;
	}
	checkCudaErrors(cudaGetLastError());
	cudaFree(d_result);
	long t4 = Util::get_cur_time();
	cerr<<"copy result used: "<<(t4-t3)<<"ms"<<endl;
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
	id_map = this->id2pos;

    delete[] qnum;
    delete[] cand;
	release();
}

void
Match::release()
{
	delete[] this->pos2id;
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
}

