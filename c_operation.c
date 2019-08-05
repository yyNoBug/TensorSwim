/*gcc -o libpycall.so -shared -fPIC pycall.c */

#include <stdio.h>
#include <stdlib.h>


void cov2d(
    int batch,
    float* i,
    int i_h,
    int i_w,
    int i_c,
    int stride_h,
    int stride_w,
    float* f,
    int f_h,
    int f_w,
    int o_c,
    float* o,
    int o_h,
    int o_w ) {




}




int fact(int n)
{
    if (n <= 1)
        return 1;
    else
        return n * fact(n - 1);
}