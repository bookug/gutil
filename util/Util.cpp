/*=============================================================================
# Filename: Util.cpp
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 17:23
# Description: 
=============================================================================*/

#include "Util.h"

using namespace std;

string Util::profile = "init.conf";
map<string, string> Util::global_config;
string Util::tmp_path = "./";

Util::Util()
{
}

Util::~Util()
{
}


//set hash table
HashFunction Util::hash[] = { Util::simpleHash, Util::APHash, Util::BKDRHash, Util::DJBHash, Util::ELFHash, \
	Util::DEKHash, Util::BPHash, Util::FNVHash, Util::HFLPHash, Util::HFHash, Util::JSHash, \
	Util::PJWHash, Util::RSHash, Util::SDBMHash, Util::StrHash, Util::TianlHash, NULL};

//remove spaces in the left
char*
Util::l_trim(char* szOutput, const char* szInput)
{
    assert(szInput != NULL);
    assert(szOutput != NULL);
    assert(szOutput != szInput);
    for   (; *szInput != '\0' && isspace(*szInput); ++szInput);
    return strcpy(szOutput, szInput);
}

//remove spaces in the right
char*
Util::r_trim(char *szOutput, const char* szInput)
{
    char *p = NULL;
    assert(szInput != NULL);
    assert(szOutput != NULL);
    assert(szOutput != szInput);
    strcpy(szOutput, szInput);
    for(p = szOutput + strlen(szOutput) - 1; p >= szOutput && isspace(*p); --p);
    *(++p) = '\0';
    return szOutput;
}

//remove spaces in the two sides
char*
Util::a_trim(char * szOutput, const char * szInput)
{
    char *p = NULL;
    assert(szInput != NULL);
    assert(szOutput != NULL);
    l_trim(szOutput, szInput);
    for   (p = szOutput + strlen(szOutput) - 1; p >= szOutput && isspace(*p); --p);
    *(++p) = '\0';
    return szOutput;
}

bool
Util::configure()
{
    const unsigned len = 505;
    char *buf, *c;
    char buf_i[len], buf_o[len];
    FILE *fp = NULL;
	char keyname[len];
	char keyval[len];

	//initialize the settings
	Util::global_config["mode"] = "single";
	//NOTICE+BETTER+TODO:use macro is better to avoid too many judging on this variable(add a DEBUG macro at the outer)
	Util::global_config["debug_level"] = "simple";
	Util::global_config["db_home"] = ".";
	Util::global_config["db_suffix"] = ".db";
	Util::global_config["buffer_maxium"] = "100";
	Util::global_config["thread_maxium"] = "1000";
	//TODO:to be recoverable
	Util::global_config["operation_logs"] = "true";

#ifdef DEBUG
	fprintf(stderr, "profile: %s\n", profile.c_str());
#endif
    if((fp = fopen(profile.c_str(), "r")) == NULL)  //NOTICE: this is not a binary file
    {
#ifdef DEBUG
        fprintf(stderr, "openfile [%s] error [%s]\n", profile.c_str(), strerror(errno));
#endif
        return false;
    }
    fseek(fp, 0, SEEK_SET);

    while(!feof(fp) && fgets(buf_i, len, fp) != NULL)
    {
		//fprintf(stderr, "buffer: %s\n", buf_i);
        Util::l_trim(buf_o, buf_i);
        if(strlen(buf_o) <= 0)
            continue;
        buf = NULL;
        buf = buf_o;
		if(buf[0] == '#')
		{
			continue;
		}
		else if(buf[0] == '[') 
		{
			continue;
		} 
		if((c = (char*)strchr(buf, '=')) == NULL)
			continue;
		memset(keyname, 0, sizeof(keyname));
		sscanf(buf, "%[^=|^ |^\t]", keyname);
#ifdef DEBUG
				//fprintf(stderr, "keyname: %s\n", keyname);
#endif
		sscanf(++c, "%[^\n]", keyval);
		char *keyval_o = (char *)calloc(strlen(keyval) + 1, sizeof(char));
		if(keyval_o != NULL) 
		{
			Util::a_trim(keyval_o, keyval);
#ifdef DEBUG
			//fprintf(stderr, "keyval: %s\n", keyval_o);
#endif
			if(keyval_o && strlen(keyval_o) > 0)
			{
				//strcpy(keyval, keyval_o);
				global_config[string(keyname)] = string(keyval_o);
			}
			xfree(keyval_o);
		}
	}

    fclose(fp);
	//display all settings here
	cout<<"the current settings are as below: "<<endl;
	cout<<"key : value"<<endl;
	cout<<"------------------------------------------------------------"<<endl;
	for(map<string, string>::iterator it = global_config.begin(); it != global_config.end(); ++it)
	{
		cout<<it->first<<" : "<<it->second<<endl;
	}
	cout<<endl;

	return true;
	//return Util::config_setting() && Util::config_debug() && Util::config_advanced();
}

long
Util::get_cur_time()
{
    timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec*1000 + tv.tv_usec/1000);
}

unsigned 
Util::RoundUp(int num, int base)
{
    unsigned tmp = (num+base-1)/base;
    return tmp * base;
}

unsigned 
Util::RoundUpDivision(int num, int base)
{
    return (num+base-1)/base;
}

//the seed is a prime, which can be well chosed to yield good performance(low conflicts)
uint32_t 
Util::MurmurHash2(const void * key, int len, uint32_t seed) 
{
    // 'm' and 'r' are mixing constants generated offline.
    // They're not really 'magic', they just happen to work well.
    const uint32_t m = 0x5bd1e995;
    const int r = 24;
    // Initialize the hash to a 'random' value
    uint32_t h = seed ^ len;
    // Mix 4 bytes at a time into the hash
    const unsigned char * data = (const unsigned char *) key;
    while (len >= 4) 
    {
        uint32_t k = *(uint32_t*) data;
        k *= m;
        k ^= k >> r;
        k *= m;
        h *= m;
        h ^= k;
        data += 4;
        len -= 4;
    }
    // Handle the last few bytes of the input array
    switch (len) 
    {
        case 3:
            h ^= data[2] << 16;
        case 2:
            h ^= data[1] << 8;
        case 1:
          h ^= data[0];
          h *= m;
    };
    // Do a few final mixes of the hash to ensure the last few
    // bytes are well-incorporated.
    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;
    return h;
}

void 
Util::DisplayBinary(int num)
{
    int i, j;
    j = sizeof(int)*8;
    for (i = 0; i < j; i ++)
    {
        //NOTICE:the most significant bit is output first
        //&0x1 is needed because for negative integers the right shift will add 1 in the former bits
        printf("%d", (num>>(j-i-1)&0x1));
    }
}

//NOTICE: we had better watch the virtual memory usage, i.e., vsize.
//https://blog.csdn.net/zyboy2000/article/details/50456764
//The unit of return values are both KB.
void 
Util::process_mem_usage(double& vm_usage, double& resident_set)
{
   using std::ios_base;
   using std::ifstream;
   using std::string;

   vm_usage     = 0.0;
   resident_set = 0.0;

   // 'file' stat seems to give the most reliable results
   ifstream stat_stream("/proc/self/stat",ios_base::in);

   // dummy vars for leading entries in stat that we don't care about
   string pid, comm, state, ppid, pgrp, session, tty_nr;
   string tpgid, flags, minflt, cminflt, majflt, cmajflt;
   string utime, stime, cutime, cstime, priority, nice;
   string O, itrealvalue, starttime;

   // the two fields we want
   unsigned long vsize;  // for virtual memory, the virtual size (B)
   long rss;    //for physical memory, number of real pages
   //In /proc/$pid/status, these corresponding to VmSize and VmRSS

   stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
               >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
               >> utime >> stime >> cutime >> cstime >> priority >> nice
               >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

   stat_stream.close();

   long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
   vm_usage     = vsize / 1024.0;
   resident_set = rss * page_size_kb;
}

string 
Util::getThreadID()
{
	//thread::id, neither int or long
	auto myid = this_thread::get_id();
	stringstream ss;
	ss << myid;
	return ss.str();
}

int
Util::memUsedPercentage()
{
    FILE* fp = fopen("/proc/meminfo", "r");
    if(fp == NULL)
        return -1;
    char str[20], tail[3];
    unsigned t, sum, used = 0;		//WARN:unsigned,memory cant be too large!
    fscanf(fp, "%s%u%s", str, &sum, tail);       //MemTotal, KB
    fscanf(fp, "%s%u%s", str, &used, tail);		//MemFree
    fscanf(fp, "%s%u%s", str, &t, tail);
    if(strcmp(str, "MemAvailable") == 0)
    {
        //QUERY:what is the relation between MemFree and MemAvailable?
        used = t;
        //scanf("%s%u%s", str, &t, tail);		//Buffers
        //used += t;
        //scanf("%s%u%s", str, &t, tail);		//Cached
        //used += t;
    }
    //else							//Buffers
    //{
    //	scanf("%s%u%s", str, &t, tail);		//Cached
    //	used += t;
    //}
    used = sum - used;
    fclose(fp);
    return (int)(used * 100.0 / sum);
}

int
Util::memoryLeft()
{
    FILE* fp = fopen("/proc/meminfo", "r");
    if(fp == NULL)
        return -1;

    char str[20], tail[3];
    unsigned num, avail = 0, free = 0, buffer = 0, cache = 0;		//WARN:unsigned,memory cant be too large!
    while (fscanf(fp, "%s%u%s", str, &num, tail) != EOF)
    {
        if(strcmp(str, "MemAvailable:") == 0)
        	avail = num;
        if(strcmp(str, "MemFree:") == 0)
        	free = num;
        if(strcmp(str, "Buffers:") == 0)
        	buffer = num;
        if(strcmp(str, "Cached:") == 0)
        	cache = num;
    }

    if (avail == 0)
    	avail = free + buffer + cache;

    fclose(fp);
    return avail / Util::MB;
}

//NOTICE: require that the list is ordered
unsigned
Util::removeDuplicate(unsigned* _list, unsigned _len)
{
	if (_list == NULL || _len == 0) {
		return 0;
	}
	unsigned valid = 0, limit = _len - 1;
	for(unsigned i = 0; i < limit; ++i)
	{
		if(_list[i] != _list[i+1])
		{
			_list[valid++] = _list[i];
		}
	}
	_list[valid++] = _list[limit];

	return valid;
}

int
Util::cmp_int(const void* _i1, const void* _i2)
{
    return *(int*)_i1  -  *(int*)_i2;
}

int
Util::cmp_unsigned(const void* _i1, const void* _i2)
{
	unsigned t1 = *(unsigned*)_i1;
	unsigned t2 = *(unsigned*)_i2;
	if(t1 > t2)
	{
		return 1;
	}
	else if(t1 == t2)
	{
		return 0;
	}
	else //t1 < t2
	{
		return -1;
	}
}

void
Util::sort(unsigned*& _id_list, unsigned _list_len)
{
#ifndef PARALLEL_SORT
	qsort(_id_list, _list_len, sizeof(unsigned), Util::cmp_unsigned);
#else
    omp_set_num_threads(thread_num);
    __gnu_parallel::sort(_id_list, _id_list + _list_len, Util::parallel_cmp_unsigned);
#endif
}

unsigned
Util::bsearch_int_uporder(unsigned _key, const unsigned* _array, unsigned _array_num)
{
    if (_array_num == 0)
    {
        //return -1;
		return INVALID;
    }
    if (_array == NULL)
    {
        //return -1;
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
        //return -1;
		return INVALID;
    }

    unsigned low = 0;
    unsigned high = _array_num - 1;

    unsigned mid;
    while (low <= high)
    {
        mid = (high - low) / 2 + low;
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

    //return -1;
	return INVALID;
}

unsigned
Util::bsearch_vec_uporder(unsigned _key, const vector<unsigned>* _vec)
{
    unsigned tmp_size = _vec->size();
    if (tmp_size == 0)
    {
        //return -1;
		return INVALID;
    }

    unsigned _first = (*_vec)[0];
    unsigned _last = (*_vec)[tmp_size - 1];

    if (_key == _last)
    {
        return tmp_size - 1;
    }

    bool not_find = (_last < _key || _first > _key);
    if (not_find)
    {
        //return -1;
		return INVALID;
    }

    unsigned low = 0;
    unsigned high = tmp_size - 1;
    unsigned mid;

    while (low <= high)
    {
        mid = (high - low) / 2 + low;
        if ((*_vec)[mid] == _key)
        {
            return mid;
        }

        if ((*_vec)[mid] > _key)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
        }
    }

    //return -1;
	return INVALID;
}

string
Util::result_id_str(vector<unsigned*>& _v, int _var_num)
{
    stringstream _ss;

	unsigned size = _v.size();
    for(unsigned i = 0; i < size; ++i)
    {
        unsigned* _p_int = _v[i];
        _ss << "[";
        for(int j = 0; j < _var_num-1; j ++)
        {
            _ss << _p_int[j] << ",";
        }
        _ss << _p_int[_var_num-1] << "]\t";
    }

    return _ss.str();
}


bool
Util::dir_exist(const string _dir)
{
	DIR* dirptr = opendir(_dir.c_str());
	if(dirptr != NULL)
	{
		closedir(dirptr);
		return true;
	}

	return false;
}

bool
Util::create_dir(const  string _dir)
{
    if(! Util::dir_exist(_dir))
    {
        mkdir(_dir.c_str(), 0755);
        return true;
    }

    return false;
}

bool
Util::create_file(const string _file) {
	if (creat(_file.c_str(), 0755) > 0) {
		return true;
	}
	return false;
}

string 
Util::get_date_time()
{
	time_t timep;
	time(&timep);
	char tmp[64];
	strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S",localtime(&timep) );
	return tmp;
}

bool
Util::save_to_file(const char* _dir, const string _content)
{
    ofstream fout(_dir);

    if (fout.is_open())
    {
        fout << _content;
        fout.close();
    }

    return false;
}

int
Util::compare(const char* _str1, unsigned _len1, const char* _str2, unsigned _len2)
{
    int ifswap = 1;		//1 indicate: not swapped
    if(_len1 > _len2)
    {
        const char* str = _str1;
        _str1 = _str2;
        _str2 = str;
        unsigned len = _len1;
        _len1 = _len2;
        _len2 = len;
        ifswap = -1;
    }
    unsigned i;
    //DEBUG: if char can be negative, which cause problem when comparing(128+)
    //
    //NOTICE:little-endian-storage, when string buffer poniter is changed to
    //unsigned long long*, the first char is the lowest byte!
    /*
       unsigned long long *p1 = (unsigned long long*)_str1, *p2 = (unsigned long long*)_str2;
       unsigned limit = _len1/8;
       for(i = 0; i < limit; ++i, ++p1, ++p2)
       {
       if((*p1 ^ *p2) == 0)	continue;
       else
       {
       if(*p1 < *p2)	return -1 * ifswap;
       else			return 1 * ifswap;
       }
       }
       for(i = 8 * limit; i < _len1; ++i)
       {
       if(_str1[i] < _str2[i])	return -1 * ifswap;
       else if(_str1[i] > _str2[i])	return 1 * ifswap;
       else continue;
       }
       if(i == _len2)	return 0;
       else	return -1 * ifswap;
       */
    for(i = 0; i < _len1; ++i)
    {   //ASCII: 0~127 but c: 0~255(-1) all transfered to unsigned char when comparing
        if((unsigned char)_str1[i] < (unsigned char)_str2[i])
            return -1 * ifswap;
        else if((unsigned char)_str1[i] > (unsigned char)_str2[i])
            return 1 * ifswap;
        else;
    }
    if(i == _len2)
        return 0;
    else
        return -1 * ifswap;
}

int
Util::string2int(string s)
{
    return atoi(s.c_str());
}

string
Util::int2string(long n)
{
    string s;
    stringstream ss;
    ss<<n;
    ss>>s;
    return s;
}

//NOTICE: there does not exist itoa() function in Linux, atoi() is included in stdlib.h
//itoa() is not a standard C function, and it is only used in Windows.
//However, there do exist a function called sprintf() in standard library which can replace itoa()
//char str[255];
//sprintf(str, "%x", 100); //将100转为16进制表示的字符串
char* 
Util::itoa(int num, char* str, int radix) //the last parameter means the number's radix: decimal, or octal formats
{
	//index table
	char index[]="0123456789ABCDEF";
	unsigned unum;
	int i=0,j,k;
	if(radix==10&&num<0)  //negative in decimal
	{
		unum=(unsigned)-num;
		str[i++]='-';
	}
	else unum=(unsigned)num;
	do{
		str[i++]=index[unum%(unsigned)radix];
		unum/=radix;
	}while(unum);
	str[i]='\0';
	//reverse order
	if(str[0]=='-')k=1;
	else k=0;
	char temp;
	for(j=k;j<=(i-1)/2;j++)
	{
		temp = str[j];
		str[j] = str[i-1+k-j];
		str[i-1+k-j] = temp;
	}
	return str;
}


string
Util::showtime()
{
//	fputs("\n\n", logsfp);
    time_t now;
    time(&now);
//	fputs(ctime(&now), logsfp);
    return string("\n\n") + ctime(&now);
}

string
Util::getQueryFromFile(const char* _file_path)
{
#ifdef DEBUG_PRECISE
    cerr << "file to open: " << _file_path <<endl;
#endif
    char buf[10000];
    std::string query_file;

    ifstream fin(_file_path);
    if(!fin)
    {
        printf("can not open: %s\n", _file_path);
        return "";
    }

    memset(buf, 0, sizeof(buf));
    stringstream _ss;
    while(!fin.eof())
    {
        fin.getline(buf, 9999);
        _ss << buf << "\n";
    }
    fin.close();

    return _ss.str();
}

string
Util::getItemsFromDir(string _path)
{
	DIR* dp = NULL;
	struct dirent* entry;
	string ret = "";
	if((dp = opendir(_path.c_str())) == NULL)
	{
		fprintf(stderr, "error opening directory!\n");
	}
	else
	{
		while((entry = readdir(dp)) != NULL)
		{
#ifdef DEBUG_PRECISE
			fprintf(stderr, "%s\n", entry->d_name);
#endif
			string name= string(entry->d_name);
			int len = name.length();
			if(len <= 3)
			{
				continue;
			}

			//BETTER: change to use db_suffix instead
			if(name.substr(len-3, 3) == ".db")
			{
				string tmp = name.substr(0, len-3);
				if(ret == "")
					ret = tmp;
				else
					ret = ret + "  " + tmp;
			}
		}
		closedir(dp);
	}
#ifdef DEBUG_PRECISE
	fprintf(stderr, "OUT getItemsFromDir\n");
#endif
	return ret;
}

//NOTICE: system() is implemented by fork() and exec(), the latter will change the whole control flow, 
//so it must be used with fork()
//The fork() will copy the whole process's image, including the heap and stack.
//But in UNIX, this copy is only logical, in practice it is copy-on-write, which will save a lot memory usage.
//Another function is vfork(), which won't copy a lot of things
//
//http://www.cnblogs.com/wuchanming/p/3784862.html
//http://www.cnblogs.com/sky-heaven/p/4687489.html
//
//BETTER: sysytem() may have some risks, using popen() instead?
string
Util::getSystemOutput(string cmd)
{
    string ans = "";
    string file = Util::tmp_path;
    file += "ans.txt";
    cmd += " > ";
    cmd += file;
    //cerr << cmd << endl;
    int ret = system(cmd.c_str());
    cmd = "rm -rf " + file;
    if(ret < 0)
    {
        fprintf(stderr, "system call failed!\n");
        system(cmd.c_str());
        return NULL;
    }

    ifstream fin(file.c_str());
    if(!fin)
    {
        cerr << "getSystemOutput: Fail to open : " << file << endl;
        return NULL;
    }

	string temp;
	getline(fin, temp);
	while(!fin.eof())
	{
		//cout<<"system line"<<endl;
		if(ans == "")
			ans = temp;
		else
			ans = ans + "\n" + temp;
		getline(fin, temp);
	}
    fin.close();
    //FILE *fp = NULL;
    //if((fp = fopen(file.c_str(), "r")) == NULL)
    //{
    //fprintf(stderr, "unbale to open file: %s\n", file.c_str());
    //}
    //else
    //{
    //char *ans = (char *)malloc(100);
    //fgets(path, 100, fp);
    //char *find = strchr(path, '\n');
    //if(find != NULL)
    //*find = '\0';
    //fclose(fp);
    //}
    system(cmd.c_str());
	//cerr<<"ans: "<<ans<<endl;
    return ans;
}

// Get the exact file path from given string: ~, ., symbol links
string
Util::getExactPath(const char *str)
{
    string cmd = "realpath ";
    cmd += string(str);

    return getSystemOutput(cmd);
}

unsigned
Util::BKDRHash(const char *_str)
{
    unsigned int seed = 131; // 31 131 1313 13131 131313 etc..
    unsigned int key = 0;

    //for(unsigned i = 0; i < i; ++i)
	while(*_str)
    {
        //key = key * seed + _str[i];
		key = key * seed + *(_str++);
    }

    return (key & 0x7FFFFFFF);
}

unsigned
Util::simpleHash(const char *_str)
{
    unsigned int key;
    unsigned char *p;

    for(key = 0, p = (unsigned char *)_str; *p; p++)
        key = 31 * key + *p;

    return (key & 0x7FFFFFFF);
}

unsigned
Util::RSHash(const char *_str)
{
    unsigned int b = 378551;
    unsigned int a = 63689;
    unsigned int key = 0;

    while (*_str)
    {
        key = key * a + (*_str++);
        a *= b;
    }

    return (key & 0x7FFFFFFF);
}

unsigned
Util::JSHash(const char *_str)
{
    unsigned int key = 1315423911;

    while (*_str)
    {
        key ^= ((key << 5) + (*_str++) + (key >> 2));
    }

    return (key & 0x7FFFFFFF);
}

unsigned
Util::PJWHash(const char *_str)
{
    unsigned int bits_in_unsigned_int = (unsigned int)(sizeof(unsigned int) * 8);
    unsigned int three_quarters = (unsigned int)((bits_in_unsigned_int * 3) / 4);
    unsigned int one_eighth = (unsigned int)(bits_in_unsigned_int / 8);

    unsigned int high_bits = (unsigned int)(0xFFFFFFFF) << (bits_in_unsigned_int - one_eighth);
    unsigned int key = 0;
    unsigned int test = 0;

    while (*_str)
    {
        key = (key << one_eighth) + (*_str++);
        if ((test = key & high_bits) != 0)
        {
            key = ((key ^ (test >> three_quarters)) & (~high_bits));
        }
    }

    return (key & 0x7FFFFFFF);
}

unsigned
Util::ELFHash(const char *_str)
{
    unsigned int key = 0;
    unsigned int x  = 0;

    while (*_str)
    {
        key = (key << 4) + (*_str++);
        if ((x = key & 0xF0000000L) != 0)
        {
            key ^= (x >> 24);
            key &= ~x;
        }
    }

    return (key & 0x7FFFFFFF);
}

unsigned
Util::SDBMHash(const char *_str)
{
    unsigned int key = 0;

    while (*_str)
    {
        key = (*_str++) + (key << 6) + (key << 16) - key;
    }

    return (key & 0x7FFFFFFF);
}

unsigned
Util::DJBHash(const char *_str)
{
    unsigned int key = 5381;
    while (*_str) {
        key += (key << 5) + (*_str++);
    }
    return (key & 0x7FFFFFFF);
}

unsigned
Util::APHash(const char *_str)
{
    unsigned int key = 0;
    int i;

    for (i=0; *_str; i++)
    {
        if ((i & 1) == 0)
        {
            key ^= ((key << 7) ^ (*_str++) ^ (key >> 3));
        }
        else
        {
            key ^= (~((key << 11) ^ (*_str++) ^ (key >> 5)));
        }
    }

    return (key & 0x7FFFFFFF);
}

unsigned
Util::DEKHash(const char* _str)
{
    unsigned int hash = strlen(_str);
    for(; *_str; _str++) 
	{
        hash = ((hash << 5) ^ (hash >> 27)) ^ (*_str);
    }
    return hash;
}

unsigned
Util::BPHash(const char* _str)
{
    unsigned int hash = 0;
    for(; *_str; _str++) 
	{
        hash = hash << 7 ^ (*_str);
    }

    return hash;
}

unsigned
Util::FNVHash(const char* _str)
{
    const unsigned int fnv_prime = 0x811C9DC5;
    unsigned int hash = 0;

    for(; *_str; _str++) 
	{
        hash *= fnv_prime;
        hash ^= (*_str);
    }

    return hash;
}

unsigned
Util::HFLPHash(const char* _str)
{
    unsigned int n = 0;
    char* b = (char*)&n;
    unsigned int len = strlen(_str);
    for(unsigned i = 0; i < len; ++i)
    {
        b[i%4] ^= _str[i];
    }
    return n%len;
}

unsigned
Util::HFHash(const char* _str)
{
    int result=0;
    const char* ptr = _str;
    int c;
    unsigned int len = strlen(_str);
    for(int i=1; (c=*ptr++); i++)
        result += c*3*i;
    if (result<0)
        result = -result;
    return result%len;
}

unsigned
Util::StrHash(const char* _str)
{
    register unsigned int   h;
    register unsigned char *p;
    for(h = 0, p = (unsigned char *)_str; *p; p++) 
	{
        h = 31 * h + *p;
    }

    return h;

}

unsigned
Util::TianlHash(const char* _str)
{
    unsigned long urlHashValue=0;
    int ilength=strlen(_str);
    int i;
    unsigned char ucChar;
    if(!ilength)  {
        return 0;
    }
    if(ilength<=256)  {
        urlHashValue=16777216*(ilength-1);
    } else {
        urlHashValue = 42781900080;
    }
    if(ilength<=96) {
        for(i=1; i<=ilength; i++) {
            ucChar = _str[i-1];
            if(ucChar<='Z'&&ucChar>='A')  {
                ucChar=ucChar+32;
            }
            urlHashValue+=(3*i*ucChar*ucChar+5*i*ucChar+7*i+11*ucChar)%1677216;
        }
    } else  {
        for(i=1; i<=96; i++)
        {
            ucChar = _str[i+ilength-96-1];
            if(ucChar<='Z'&&ucChar>='A')
            {
                ucChar=ucChar+32;
            }
            urlHashValue+=(3*i*ucChar*ucChar+5*i*ucChar+7*i+11*ucChar)%1677216;
        }
    }

    return urlHashValue;
}

//NOTICE: the time of log() and sqrt() in C can be seen as constant

//NOTICE:_b must >= 1
double
Util::logarithm(double _a, double _b)
{
    //REFRENCE: http://blog.csdn.net/liyuanbhu/article/details/8997850
    //a>0 != 1; b>0 (b>=2 using log/log10/change, 1<b<2 using log1p, b<=1?)
    if(_a <= 1 || _b < 1)
        return -1.0;
    double under = log(_a);
    if(_b == 1)
        return 0.0;
    else if(_b < 2)
        return log1p(_b - 1) / under;
    else //_b >= 2
        return log(_b) / under;
    return -1.0;
}


void
Util::intersect(unsigned*& _id_list, unsigned& _id_list_len, const unsigned* _list1, unsigned _len1, const unsigned* _list2, unsigned _len2)
{
	vector<unsigned> res;
	//cout<<"intersect prevar: "<<_len1<<"   "<<_len2<<endl;
	if(_list1 == NULL || _len1 == 0 || _list2 == NULL || _len2 == 0)
	{
		_id_list = NULL;
		_id_list_len = 0;
	}

	//when size is almost the same, intersect O(n)
	//when one size is small ratio, search in the larger one O(mlogn)
	//
	//n>0 m=nk(0<k<1) 
	//compare n(k+1) and nklogn: k0 = log(n/2)2 requiring that n>2
	//k<=k0 binary search; k>k0 intersect
	int method = -1; //0: intersect 1: search in list1 2: search in list2
	unsigned n = _len1;
	double k = 0;
	if(n < _len2)
	{
		k = (double)n / (double)_len2;
		n = _len2;
		method = 2;
	}
	else
	{
		k = (double)_len2 / (double)n;
		method = 1;
	}
	if(n <= 2)
		method = 0;
	else
	{
		double limit = Util::logarithm(n/2, 2);
		if(k > limit)
			method = 0;
	}

	switch(method)
	{
	case 0:
	{   //this bracket is needed if vars are defined in case
		unsigned id_i = 0;
		unsigned num = _len1;
		for(unsigned i = 0; i < num; ++i)
		{
			unsigned can_id = _list1[i];
			while((id_i < _len2) && (_list2[id_i] < can_id))
			{
				id_i ++;
			}

			if(id_i == _len2)
			{
				break;
			}

			if(can_id == _list2[id_i])
			{
				res.push_back(can_id);
				id_i ++;
			}
		}
		break;
	}
	case 1:
	{
		for(unsigned i = 0; i < _len2; ++i)
		{
			if(Util::bsearch_int_uporder(_list2[i], _list1, _len1) != INVALID)
				res.push_back(_list2[i]);
		}
		break;
	}
	case 2:
	{
		unsigned m = _len1, i;
		for(i = 0; i < m; ++i)
		{
			unsigned t = _list1[i];
			if(Util::bsearch_int_uporder(t, _list2, _len2) != INVALID)
				res.push_back(t);
		}
		break;
	}
	default:
		cerr << "no such method in Util::intersect()" << endl;
		break;
	}

	_id_list_len = res.size();

	if (_id_list_len == 0) {
		_id_list = NULL;
	}
	else {
		_id_list = new unsigned[_id_list_len];
		for (unsigned i = 0; i < _id_list_len; ++i)
			_id_list[i] = res[i];
	}
	delete[] _list1;
	delete[] _list2;
}

int
Util::compIIpair(int _a1, int _b1, int _a2, int _b2)
{
	if(_a1 == _a2 && _b1 == _b2)
		return 0;
	else if(_a1 < _a2 || (_a1 == _a2 && _b1 <= _b2))
		return -1;
	else
		return 1;
}

bool 
Util::isValidPort(string str) 
{
	//valid port number: 0 - 65535
	if(str.length() < 1 || str.length() > 5) 
	{
		return false;
	}

	unsigned i;
	for(i = 0; i < str.length(); i++) 
	{
		if(str[i] < '0' || str[i] > '9') 
		{
			return false;
		}
	}

	int port = Util::string2int(str);
	if(port < 0 || port>65535) 
	{
		return false;
	}

	return true;
}

bool 
Util::isValidIP(string str) 
{
	if(str == "localhost") 
	{
		return true;
	}
	return (Util::isValidIPV4(str) || Util::isValidIPV6(str));
}

bool 
Util::isValidIPV4(string str) 
{
	regex_t reg;
	char pattern[] = "^(([01]?[0-9][0-9]?|2[0-4][0-9]|25[0-5])\\.){3}([01]?[0-9][0-9]?|2[0-4][0-9]|25[0-5])$";
	regcomp(&reg, pattern, REG_EXTENDED | REG_NOSUB);
	regmatch_t pm[1];
	int status = regexec(&reg, str.c_str(), 1, pm, 0);
	regfree(&reg);
	if(status == REG_NOMATCH) 
	{
		return false;
	}
	return true;
}

bool 
Util::isValidIPV6(string str) 
{
	//TO BE IMPLEMENTED
	return false;
}

string 
Util::getTimeName()
{
	//NOTICE: this is another method to get the concrete time
	time_t rawtime;
	struct tm* timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	string tempTime = asctime(timeinfo);
	for(int i = 0; i < tempTime.length(); i++)
	{
		if(tempTime[i] == ' ')
			tempTime[i] = '_';
	}
	string myTime = tempTime.substr(0, tempTime.length()-1);
	return myTime;
}

string
Util::getTimeString() {
	static const int max = 20; // max length of time string
	char time_str[max];
	time_t timep;
	time(&timep);
	strftime(time_str, max, "%Y%m%d %H:%M:%S\t", gmtime(&timep));
	return string(time_str);
}

//is ostream.write() ok to update to disk at once? all add ofstream.flush()?
//http://bookug.cc/rwbuffer
//BETTER: add a sync function in Util to support FILE*, fd, and fstream
void 
Util::Csync(FILE* _fp)
{
	//NOTICE: fclose will also do fflush() operation, but not others
	if(_fp == NULL)
	{
		return; 
	}
	//this will update the buffer from user mode to kernel mode
	fflush(_fp);
	//change to Unix fd and use fsync to sync to disk: fileno(stdin)=0
	int fd = fileno(_fp);
	fsync(fd);
	//FILE * fp = fdopen (1, "w+");   //file descriptor to file pointer 
	//NOTICE: disk scheduler also has a small buffer, but there is no matter even if the power is off
	//(UPS for each server to enable the synchronization between scheduler and disk)
}

string
Util::node2string(const char* _raw_str) {
	string _output;
	unsigned _first_quote = 0;
	unsigned _last_quote = 0;
	bool _has_quote = false;
	for (unsigned i = 0; _raw_str[i] != '\0'; i++) {
		if (_raw_str[i] == '\"') {
			if (!_has_quote) {
				_first_quote = i;
				_last_quote = i;
				_has_quote = true;
			}
			else {
				_last_quote = i;
			}
		}
	}
	if (_first_quote==_last_quote) {
		_output += _raw_str;
		return _output;
	}
	for (unsigned i = 0; i <= _first_quote; i++) {
		_output += _raw_str[i];
	}
	for (unsigned i = _first_quote + 1; i < _last_quote; i++) {
		switch (_raw_str[i]) {
		case '\n':
			_output += "\\n";
			break;
		case '\r':
			_output += "\\r";
			break;
		case '\t':
			_output += "\\t";
			break;
		case '\"':
			_output += "\\\"";
			break;
		case '\\':
			_output += "\\\\";
			break;
		default:
			_output += _raw_str[i];
		}
	}
	for (unsigned i = _last_quote; _raw_str[i] != 0; i++) {
		_output += _raw_str[i];
	}
	return _output;
}

void
Util::empty_file(const char* _fname)
{
	FILE * fp;
	//NOTICE: if exist, then overwrite and create a empty file
	fp = fopen(_fname, "w"); 
	if(fp == NULL)
	{
		printf("do empty file %s failed\n", _fname);
	}
	else 
	{
		fclose(fp);
	}
}

//require that _base>=1
unsigned 
ceiling(unsigned _val, unsigned _base)
{
	//WARN: we donot check overflow here
	return (_val+_base-1) / _base * _base;
}





