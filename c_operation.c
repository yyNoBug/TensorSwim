/*gcc -o libpycall.so -shared -fPIC c_operation.c -lopenblas -O4*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>


void matmul(const float* A, const float* B, float* C, int M, int N, int K, int transA, int transB) {
    const CBLAS_TRANSPOSE TA = transA ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE TB = transB ? CblasTrans : CblasNoTrans;
    const int lda = transA ? M : N;
    const int ldb = transB ? N : K;
    const int ldc = K;
    const float aa = 1;
    const float bb = 0;
    cblas_sgemm(CblasRowMajor, TA, TB, M, K, N, aa, A, lda, B, ldb, bb, C, ldc);
}


void zero_extend(float* in, float* out, int u, int d, int l, int r,
                 int shp0, int shp1, int shp2, int shp3) {

    int shpp1 = shp1 + u + d;
    int shpp2 = shp2 + l + r;
    for (int i = 0; i < shp0; ++i) {
        for (int j = 0; j < shp1; ++j) {
            for (int k = 0; k < shp2; ++k) {
                memcpy(out + ((i * shpp1 + j + u) * shpp2 + k + l) * shp3,
                        in + ((i * shp1 + j) * shp2 + k) * shp3,
                        shp3 * sizeof(float));
            }
        }
    }
}


void conv2d(
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

    int imgsize = o_h * o_w * f_h * f_w * i_c * sizeof(float);
    float* image = (float*) malloc(imgsize);
    float* ptout = ou;

    for (int i = 0; i < batch; ++i) {
        memset(image, 0, imgsize);
        float* ptimg = image;

        for (int j = 0; j < o_h; ++j) {
            for (int k = 0; k < o_w; ++k) {
                for (int p = 0; p < f_h; ++p) {
                    for (int q = 0; q < f_w; ++q) {
                        memcpy(ptimg, in + ((i * i_h + j + p) * i_w + k + q) * i_c, i_c * sizeof(float));
                        ptimg += i_c;
                    }
                }
            }
        }
        matmul(image, ft, ou, o_h * o_w, f_h * f_w * i_c, o_c, 0, 0);
        ou += o_h * o_w * o_c;
    }

    free(image);
}


void conv2d_grad2(
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

    int imgsize = o_h * o_w * f_h * f_w * i_c * sizeof(float);
    float* image = (float*) malloc(imgsize);

    int ftsize = f_h * f_w * i_c * o_c;
    float* tmp = (float*) malloc(ftsize * sizeof(float));

    for (int i = 0; i < batch; ++i) {
        memset(image, 0, imgsize);
        float* ptimg = image;

        for (int j = 0; j < o_h; ++j) {
            for (int k = 0; k < o_w; ++k) {
                for (int p = 0; p < f_h; ++p) {
                    for (int q = 0; q < f_w; ++q) {
                        memcpy(ptimg, in + ((i * i_h + j + p) * i_w + k + q) * i_c, i_c * sizeof(float));
                        ptimg += i_c;
                    }
                }
            }
        }
        memset(tmp, 0, ftsize * sizeof(float));
        matmul(image, ou, tmp, f_h * f_w * i_c, o_h * o_w, o_c, 1, 0);
        for(int i = 0; i < ftsize; ++i) ft[i] += tmp[i];
        ou += o_h * o_w * o_c;
    }

    free(image);
    free(tmp);
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

    //printf("CONV2D GRAD1 WARNING!!\ni:%d j:%d k:%d t:%d p:%d q:%d r:%d\n", batch, o_h, o_w, i_c, f_h, f_w, o_c);

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

    //printf("CONV2D GRAD2 WARNING!!\ni:%d j:%d k:%d t:%d p:%d q:%d r:%d\n", batch, o_h, o_w, i_c, f_h, f_w, o_c);

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