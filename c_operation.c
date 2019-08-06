/*gcc -o libpycall.so -shared -fPIC c_operation.c */

#include <stdio.h>
#include <stdlib.h>


void cov2d(
    int batch,
    float* in,
    int i_h,
    int i_w,
    int i_c,
    int stride_h,
    int stride_w,
    float* ft,
    int f_h,
    int f_w,
    int o_c,
    float* ou,
    int o_h,
    int o_w ) {

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < o_h; ++j) {
            for (int k = 0; k < o_w; ++k) {
                for (int t = 0; t < i_c; ++t) {
                    for (int p = 0; p < f_h; ++p) {
                        for (int q = 0; q < f_w; ++q) {
                            for (int r = 0; r < o_c; ++r) {
                                ou[((i * o_h + j) * o_w + k) * o_c + r] +=
                                ft[((p * f_w + q) * i_c + t) * o_c + r] *
                                in[((i * i_h + j + p) * i_w + k + q) * i_c + t];
                                /*printf("ou x1:%d x2:%d x3:%d x4:%d val:%f\nft x1:%d x2:%d x3:%d x4:%d val:%f\nin x1:%d x2:%d x3:%d x4:%d val:%f\n\n",
                                    i, j, k, r, ou[((i * o_h + j) * o_w + k) * o_c + r],
                                    p, q, t, r, ft[((p * f_w + q) * i_c + t) * o_c + r],
                                    i, j + p, k + q, t, in[((i * i_h + j + p) * i_w + k + q) * i_c + t]);*/
                            }
                        }
                    }
                }
            }
        }
    }
    //for (int i = 0; i < batch * o_h * o_w * o_c; ++i) printf("%f\n", ou[i]);

    return;
}

void cov2d_grad1(
    int batch,
    float* in,
    int i_h,
    int i_w,
    int i_c,
    int stride_h,
    int stride_w,
    float* ft,
    int f_h,
    int f_w,
    int o_c,
    float* ou,
    int o_h,
    int o_w) {

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < o_h; ++j) {
            for (int k = 0; k < o_w; ++k) {
                for (int t = 0; t < i_c; ++t) {
                    for (int p = 0; p < f_h; ++p) {
                        for (int q = 0; q < f_w; ++q) {
                            for (int r = 0; r < o_c; ++r) {
                                in[((i * i_h + j + p) * i_w + k + q) * i_c + t] +=
                                ft[((p * f_w + q) * i_c + t) * o_c + r] *
                                ou[((i * o_h + j) * o_w + k) * o_c + r];
                                /*printf("ou x1:%d x2:%d x3:%d x4:%d val:%f\nft x1:%d x2:%d x3:%d x4:%d val:%f\nin x1:%d x2:%d x3:%d x4:%d val:%f\n\n",
                                    i, j, k, r, ou[((i * o_h + j) * o_w + k) * o_c + r],
                                    p, q, t, r, ft[((p * f_w + q) * i_c + t) * o_c + r],
                                    i, j + p, k + q, t, in[((i * i_h + j + p) * i_w + k + q) * i_c + t]);*/
                            }
                        }
                    }
                }
            }
        }
    }
    //for (int i = 0; i < batch * o_h * o_w * o_c; ++i) printf("%f\n", ou[i]);

    return;
}


void cov2d_grad2(
    int batch,
    float* in,
    int i_h,
    int i_w,
    int i_c,
    int stride_h,
    int stride_w,
    float* ft,
    int f_h,
    int f_w,
    int o_c,
    float* ou,
    int o_h,
    int o_w) {

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < o_h; ++j) {
            for (int k = 0; k < o_w; ++k) {
                for (int t = 0; t < i_c; ++t) {
                    for (int p = 0; p < f_h; ++p) {
                        for (int q = 0; q < f_w; ++q) {
                            for (int r = 0; r < o_c; ++r) {
                                ft[((p * f_w + q) * i_c + t) * o_c + r] +=
                                in[((i * i_h + j + p) * i_w + k + q) * i_c + t] *
                                ou[((i * o_h + j) * o_w + k) * o_c + r];
                                /* printf("ou x1:%d x2:%d x3:%d x4:%d val:%f\nft x1:%d x2:%d x3:%d x4:%d val:%f\nin x1:%d x2:%d x3:%d x4:%d val:%f\n\n",
                                    i, j, k, r, ou[((i * o_h + j) * o_w + k) * o_c + r],
                                    p, q, t, r, ft[((p * f_w + q) * i_c + t) * o_c + r],
                                    i, j + p, k + q, t, in[((i * i_h + j + p) * i_w + k + q) * i_c + t]); */
                            }
                        }
                    }
                }
            }
        }
    }
    //for (int i = 0; i < batch * o_h * o_w * o_c; ++i) printf("%f\n", ou[i]);

    return;
}

void max_pool(
    int batch,
    float* in,
    int i_h,
    int i_w,
    int i_c,
    int stride_h,
    int stride_w,
    int f_h,
    int f_w,
    int o_c,
    float* ou,
    int o_h,
    int o_w ) {

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < o_h; ++j) {
            for (int k = 0; k < o_w; ++k) {
                for (int t = 0; t < i_c; ++t) {
                    float ma = -99999.0;
                    for (int p = 0; p < f_h; ++p) {
                        for (int q = 0; q < f_w; ++q) {
                            if (in[((i * i_h + j * stride_h + p) * i_w + k * stride_w + q) * i_c + t] > ma)
                                ma = in[((i * i_h + j * stride_h + p) * i_w + k * stride_w + q) * i_c + t];
                        }
                    }
                    ou[((i * o_h + j) * o_w + k) * o_c + t] = ma;
                }
            }
        }
    }
    //for (int i = 0; i < batch * o_h * o_w * o_c; ++i) printf("%f\n", ou[i]);

    return;
}

void max_pool_grad(
    int batch,
    float* in,
    float* gradi,
    int i_h,
    int i_w,
    int i_c,
    int stride_h,
    int stride_w,
    int f_h,
    int f_w,
    int o_c,
    float* ou,
    int o_h,
    int o_w ) {

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < o_h; ++j) {
            for (int k = 0; k < o_w; ++k) {
                for (int t = 0; t < i_c; ++t) {
                    float ma = -99999.0;
                    int ind = -1;
                    for (int p = 0; p < f_h; ++p) {
                        for (int q = 0; q < f_w; ++q) {
                            if (in[((i * i_h + j * stride_h + p) * i_w + k * stride_w + q) * i_c + t] > ma) {
                                ind = ((i * i_h + j * stride_h + p) * i_w + k * stride_w + q) * i_c + t;
                                ma = in[ind];
                            }
                        }
                    }
                    if (ind < 0) printf("BOOMSHAKALAKA");
                    gradi[ind] += ou[((i * o_h + j) * o_w + k) * o_c + t];
                }
            }
        }
    }

}


int fact(int n)
{
    if (n <= 1)
        return 1;
    else
        return n * fact(n - 1);
}