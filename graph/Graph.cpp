/*=============================================================================
# Filename: Graph.cpp
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 23:01
# Description: 
=============================================================================*/

#include "Graph.h"

using namespace std;


uint32_t hash(const void * key, int len, uint32_t seed) 
{
    return Util::MurmurHash2(key, len, seed);
}


void 
Graph::addVertex(LABEL _vlb)
{
	this->vertices.push_back(Vertex(_vlb));
}

void 
Graph::addEdge(VID _from, VID _to, LABEL _elb)
{
    Edge et(_from, _to, _elb);
    this->edges.push_back(et);
	this->vertices[_from].out.push_back(Neighbor(_to, _elb, edge_num));
	this->vertices[_to].in.push_back(Neighbor(_from, _elb, edge_num));
    this->edge_num++;
}


void 
Graph::preprocessing(bool column_oriented)
{
    unsigned deg = this->countMaxDegree();
    cout<<"maximum degree: "<<deg<<endl;

    long t1 = Util::get_cur_time();
    this->transformToCSR();
    //this->buildSignature(column_oriented);
    long t2 = Util::get_cur_time();
    printf("time of preprocessing(not included in matching): %ld ms\n", t2-t1);
	//now we can release the memory of original structure 
	//this->vertices.clear();

    int i, j;
	//to construct inverse label list
	Element* elelist = new Element[this->vertex_num];
	for(i = 0; i <this->vertex_num; ++i)
	{
		elelist[i].id = i;
		elelist[i].label = this->vertex_value[i];
	}
	sort(elelist, elelist+this->vertex_num);

	int label_num = 0;
	for(i = 0; i <this->vertex_num; ++i)
	{
		if(i == 0 || elelist[i].label != elelist[i-1].label)
		{
			label_num++;
		}
	}

	this->label_num = label_num;
	this->inverse_label = new unsigned[label_num];
	this->inverse_offset = new unsigned[label_num+1];
	this->inverse_vertex = new unsigned[this->vertex_num];
	j = 0;
	for(i = 0; i <this->vertex_num; ++i)
	{
		this->inverse_vertex[i] = elelist[i].id;
		if(i == 0 || elelist[i].label != elelist[i-1].label)
		{
			this->inverse_label[j] = elelist[i].label;
			this->inverse_offset[j] = i;
			++j;
		}
	}
	this->inverse_offset[label_num] = this->vertex_num;

	delete[] elelist;
}

void
Graph::buildSignature(bool column_oriented)
{
    cout<<"build signature for a new graph"<<endl;
    //build row oriented signatures for query graph
    unsigned signum = SIGLEN / VLEN;
    unsigned tablen = this->vertex_num * signum;
    unsigned* signature_table = new unsigned[tablen];
    memset(signature_table, 0, sizeof(unsigned)*tablen);
    unsigned gnum = 240, gsize = 2;
    for(int i = 0; i < this->vertex_num; ++i)
    {
        Vertex& v = this->vertices[i];
        int pos = hash(&(v.label), 4, HASHSEED) % VLEN;
        signature_table[signum*i] = 1 << pos;
        for(int j = 0; j < v.in.size(); ++j)
        {
            Neighbor& nb = v.in[j];
            int sig[2];
            sig[0] = this->vertices[nb.vid].label;
            sig[1] = nb.elb;
            pos = hash(sig, 8, HASHSEED) % gnum;
            int a = pos / 16, b = pos % 16;
            unsigned t = signature_table[signum*i+1+a];
            unsigned c = 3 << (2*b);
            c = c & t;
            c = c >> (2*b);
            switch(c)
            {
                case 0:
                    c = 1;
                    break;
                case 1:
                    c = 3;
                    break;
                default:  //c==3
                    c = 3;
                    break;
            }
            c = c << (2*b);
            t = t | c;
            signature_table[signum*i+1+a] = t;
        }
        for(int j = 0; j < v.out.size(); ++j)
        {
            Neighbor& nb = v.out[j];
            int sig[2];
            sig[0] = this->vertices[nb.vid].label;
            sig[1] = -nb.elb;
            int pos = hash(sig, 8, HASHSEED) % gnum;
            int a = pos / 16, b = pos % 16;
            unsigned t = signature_table[signum*i+1+a];
            unsigned c = 3 << (2*b);
            c = c & t;
            c = c >> (2*b);
            switch(c)
            {
                case 0:
                    c = 1;
                    break;
                case 1:
                    c = 3;
                    break;
                default:  //c==3
                    c = 3;
                    break;
            }
            c = c << (2*b);
            t = t | c;
            signature_table[signum*i+1+a] = t;
        }
        //for(int k = 0; k < 16; ++k)
        //{
            //Util::DisplayBinary(signature_table[signum*i+k]);
            //cout<<" ";
        //}
        //cout<<endl;
    }

    if(column_oriented)
    {
        //change to column oriented for data graph
        unsigned* new_table = new unsigned[tablen];
        unsigned base = 0;
        for(int k = 0; k < 16; ++k)
        {
            for(int i = 0; i < this->vertex_num; ++i)
            {
                new_table[base++] = signature_table[signum*i+k];
            }
        }
        delete[] signature_table;
        signature_table = new_table;
        //cout<<"column oriented signature table"<<endl;
        //for(int k = 0; k < 16; ++k)
        //{
            //for(int i = 0; i < this->vertex_num; ++i)
            //{
                //Util::DisplayBinary(signature_table[this->vertex_num*k+i]);
                //cout<<" ";
            //}
            //cout<<endl;
        //}
    }

    this->signature_table = signature_table;
}

//BETTER: construct all indices using GPU, with thrust or CUB, back40computing or moderngpu
//NOTICE: for data graph, this transformation only needs to be done once, and can match 
//many query graphs later
void 
Graph::transformToCSR()
{
	this->vertex_num = this->vertices.size();
	this->vertex_value = new unsigned[this->vertex_num];
	for(int i = 0; i < this->vertex_num; ++i)
	{
		this->vertex_value[i] = this->vertices[i].label;
        //sort on label, when label is identical, sort on VID
        sort(this->vertices[i].in.begin(), this->vertices[i].in.end());
        sort(this->vertices[i].out.begin(), this->vertices[i].out.end());
    }

    int n = this->vertex_num, m = this->edge_num;
    //only builds the out csr  
    this->ro = new unsigned[n+1];
    this->ci = new unsigned[m];
    this->ev = new unsigned[m];
    int pos = 0;
	for(int i = 0; i < this->vertex_num; ++i)
    {
        vector<Neighbor>& out = this->vertices[i].out;
        ro[i] = pos;
        for(int j = 0; j < out.size(); ++j)
        {
            ci[pos+j] = out[j].vid;
            ev[pos+j] = out[j].elb;
        }
        pos += out.size();
    }
    ro[n] = pos;

    return;

    //NOTICE: the edge label begins from 1
    this->csrs_in = new PCSR[this->edgeLabelNum+1];
    this->csrs_out = new PCSR[this->edgeLabelNum+1];
    vector<unsigned>* keys_in = new vector<unsigned>[this->edgeLabelNum+1];
    vector<unsigned>* keys_out = new vector<unsigned>[this->edgeLabelNum+1];
	for(int i = 0; i < this->vertex_num; ++i)
    {
        int insize = this->vertices[i].in.size(), outsize = this->vertices[i].out.size();
        for(int j = 0; j < insize; ++j)
        {
            int vid = this->vertices[i].in[j].vid;
            int elb = this->vertices[i].in[j].elb;
            int tsize = keys_in[elb].size();
            if(tsize == 0 || keys_in[elb][tsize-1] != i)
            {
                keys_in[elb].push_back(i);
            }
            //NOTICE: we do not use C++ reference PCSR& here because it can not change(frpm p-->A to p-->B)
            PCSR* tcsr = &this->csrs_in[elb];
            tcsr->edge_num++;
        }
        for(int j = 0; j < outsize; ++j)
        {
            int vid = this->vertices[i].out[j].vid;
            int elb = this->vertices[i].out[j].elb;
            int tsize = keys_out[elb].size();
            if(tsize == 0 || keys_out[elb][tsize-1] != i)
            {
                keys_out[elb].push_back(i);
            }
            PCSR* tcsr = &this->csrs_out[elb];
            tcsr->edge_num++;
        }
    }

    for(int i = 1; i <= this->edgeLabelNum; ++i)
    {
        PCSR* tcsr = &this->csrs_in[i];
        this->buildPCSR(tcsr, keys_in[i], i, true);
        tcsr = &this->csrs_out[i];
        this->buildPCSR(tcsr, keys_out[i], i, false);
    }
    delete[] keys_in;
    delete[] keys_out;
}

void 
Graph::buildPCSR(PCSR* pcsr, vector<unsigned>& keys, int label, bool incoming)
{
    sort(keys.begin(), keys.end());
    unsigned key_num = keys.size();
    unsigned* key_array = new unsigned[key_num ];
    unsigned* row_offset = new unsigned[key_num+1 ];
    unsigned edge_num = pcsr->edge_num;
    unsigned* column_index = new unsigned[edge_num];
    pcsr->key_num = key_num;
    pcsr->key_array = key_array;
    pcsr->row_offset = row_offset;
    pcsr->column_index = column_index;

    //copy elements to column index and set offsets
    unsigned pos = 0;
    for(int i = 0; i < key_num; ++i)
    {
        unsigned id = keys[i];
        key_array[i] = id;
        vector<Neighbor>* adjs = &this->vertices[id].out;
        if(incoming)
        {
            adjs = &this->vertices[id].in;
        }
        row_offset[i] = pos;
        for(int k = 0; k < adjs->size(); ++k)
        {
            if((*adjs)[k].elb == label)
            {
                column_index[pos++] = (*adjs)[k].vid;
            }
        }
    }
    row_offset[key_num] = pos;
}

unsigned
Graph::countMaxDegree()
{
    //BETTER: count the degree based on direction and edge labels
    int size = this->vertices.size();
    unsigned maxv = 0;
    for(int i = 0; i < size; ++i)
    {
        unsigned t = vertices[i].in.size() + vertices[i].out.size();
        if(t > maxv)
        {
            maxv = t;
        }
    }
    return maxv;
}

void 
Graph::printGraph()
{
	int i, n = this->vertex_num;
	cout<<"vertex value:"<<endl;
	for(i = 0; i < n; ++i)
	{
		cout<<this->vertex_value[i]<<" ";
	}cout<<endl;
}

