/*=============================================================================
# Filename: util.h
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 17:20
# Description: 
=============================================================================*/

#ifndef _UTIL_UTIL_H
#define _UTIL_UTIL_H

//basic macros and types are defined here, including common headers 

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
//DBL_MAX is contained in the header below
#include <float.h>
#include <dirent.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <time.h>
#include <fcntl.h>
#include <errno.h>
#include <regex.h>
#include <locale.h>
#include <assert.h>
#include <libgen.h>
#include <signal.h>

#include <sys/time.h>
#include <sys/types.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>

//NOTICE:below are restricted to C++, C files should not include(maybe nested) this header!
#include <bitset>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include <map>
#include <set>
#include <stack>
#include <queue>
#include <deque>
#include <vector>
#include <list>
#include <iterator>
#include <algorithm>
#include <functional>
#include <utility>
#include <new>

//NOTICE:below are libraries need to link
#include <thread>    //only for c++11 or greater versions
#include <atomic> 
#include <mutex> 
#include <condition_variable> 
#include <future> 
#include <memory> 
#include <stdexcept> 
#include <pthread.h> 
#include <math.h>
#include <readline/readline.h>
#include <readline/history.h>

//indicate that in debug mode
//#define DEBUG   1
#ifdef DEBUG2
#ifndef DEBUG
//#define DEBUG
#endif
#endif


#define xfree(x) free(x); x = NULL;

typedef unsigned(*HashFunction)(const char*);
//NOTICE:hash functions for int are not so many, so we represent int by a 4-byte stringinstead
//(not totally change int to string, which is costly)
//http://www.cppblog.com/aurain/archive/2010/07/06/119463.html
//http://blog.csdn.net/mycomputerxiaomei/article/details/7641221
//http://kb.cnblogs.com/page/189480/

#define MAX_PATTERN_SIZE 1000000000

//for signatures
#define SIGLEN 64*8
#define VLEN 32
#define SIGNUM SIGLEN/VLEN
#define SIGBYTE sizeof(unsigned)*SIGNUM
#define HASHSEED 17
#define HASHSEED2 53


#define xfree(x) free(x); x = NULL;

typedef int LABEL;
typedef int VID;
typedef int EID;
typedef int GID;
typedef long PID;
typedef long LENGTH;

static const unsigned INVALID = UINT_MAX;

/******** all static&universal constants and fucntions ********/
class Util
{
public:
	static const unsigned MB = 1048576;
	static const unsigned GB = 1073741824;
	static std::string getThreadID();
	static int memUsedPercentage();
	static int memoryLeft();
	static int compare(const char* _str1, unsigned _len1, const char* _str2, unsigned _len2); //QUERY(how to use default args)
	static int string2int(std::string s);
	static std::string int2string(long n);
	static char* itoa(int num, char* str, int radix);
	//string2str: s.c_str()
	//str2string: string(str)
	static int compIIpair(int _a1, int _b1, int _a2, int _b2);
	static std::string showtime();
	static int cmp_int(const void* _i1, const void* _i2);
	static int cmp_unsigned(const void* _i1, const void* _i2);
	static bool parallel_cmp_unsigned(unsigned _i1, unsigned _i2);
	static void sort(unsigned*& _id_list, unsigned _list_len);
	static unsigned bsearch_int_uporder(unsigned _key, const unsigned* _array, unsigned _array_num);
	static unsigned bsearch_vec_uporder(unsigned _key, const std::vector<unsigned>* _vec);
	static std::string result_id_str(std::vector<unsigned*>& _v, int _var_num);
	static bool dir_exist(const std::string _dir);
	static bool create_dir(const std:: string _dir);
	static bool create_file(const std::string _file);

	static std::string getTimeName();
	static std::string getTimeString();
	static long get_cur_time();
	static std::string get_date_time();
	static bool save_to_file(const char*, const std::string _content);
	static bool isValidPort(std::string);
	static bool isValidIP(std::string);
	static std::string node2string(const char* _raw_str);

	static unsigned removeDuplicate(unsigned*, unsigned);
	static void Csync(FILE* _fp);

	static std::string getQueryFromFile(const char* _file_path); 
	static std::string getSystemOutput(std::string cmd);
	static std::string getExactPath(const char* path);
	static std::string getItemsFromDir(std::string path);
	static void logging(std::string _str);
	static void empty_file(const char* _fname);
    static void DisplayBinary(int num);
    static void process_mem_usage(double& vm_usage, double& resident_set);
	static unsigned ceiling(unsigned _val, unsigned _base);
    static unsigned RoundUp(int num, int base);
    static unsigned RoundUpDivision(int num, int base);

	// Below are some useful hash functions for string
	static unsigned simpleHash(const char *_str);
	static unsigned APHash(const char *_str);
	static unsigned BKDRHash(const char *_str);
	static unsigned DJBHash(const char *_str);
	static unsigned ELFHash(const char *_str);
	static unsigned DEKHash(const char* _str);
	static unsigned BPHash(const char* _str);
	static unsigned FNVHash(const char* _str);
	static unsigned HFLPHash(const char* _str);
	static unsigned HFHash(const char* _str);
	static unsigned JSHash(const char *_str);
	static unsigned PJWHash(const char *_str);
	static unsigned RSHash(const char *_str);
	static unsigned SDBMHash(const char *_str);
	static unsigned StrHash(const char* _str);
	static unsigned TianlHash(const char* _str);
    static uint32_t MurmurHash2(const void * key, int len, uint32_t seed) ;

	static const unsigned HashNum = 16;
	static HashFunction hash[];

	static double logarithm(double _a, double _b);
	static void intersect(unsigned*& _id_list, unsigned& _id_list_len, const unsigned* _list1, unsigned _len1, const unsigned* _list2, unsigned _len2);

    static bool configure();
	static char* l_trim(char *szOutput, const char *szInput);
	static char* r_trim(char *szOutput, const char *szInput);
	static char* a_trim(char *szOutput, const char * szInput);
	static bool isValidIPV4(std::string);
	static bool isValidIPV6(std::string);

	Util();
	~Util();

private:
    static std::string tmp_path;
	static std::map<std::string, std::string> global_config;
	static std::string profile;
};

#endif //_UTIL_UTIL_H

