/*=============================================================================
# Filename: init.h
# Author: bookug
# Mail: bookug@qq.com
# Last Modified: 2019-08-30 22:55
# Description: This file is specially used by the main program, thus it only contains the initGPU function. 
It can not contain CUDA syntaxes, otherwise the compiling will fail.
=============================================================================*/

#ifndef _GUTIL_INIT_H
#define _GUTIL_INIT_H

#include "../util/Util.h" 

using namespace std; 


class gutil
{
public:
    static void initGPU(int dev);
};

#endif //_GUTIL_INIT_H




