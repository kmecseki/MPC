#pragma once
#ifndef MPC_KERNELS_H
#define MPC_KERNELS_H
#include "cuda_runtime.h"
#include "settings.h"

__global__ void memsetCUDA(double* array, size_t size);
__global__ void spatintegralCUDA(const cufftDoubleComplex *C, double *result, int *dims);
__device__ __forceinline__  double cexpCUDAr(cuDoubleComplex z);
__device__ __forceinline__  double cexpCUDAi(cuDoubleComplex z);
__global__ void calcexpD0CUDA(double dzz, double alpha, double *betas, cuDoubleComplex *exp_D0, int dims);
__global__ void propagT(cuDoubleComplex *A, const cuDoubleComplex *exponent, int *dims);
__global__ void propagR(cuDoubleComplex *A, const cuDoubleComplex *exponent, int *dims, int dir);
__global__ void calcdACUDA(cuDoubleComplex *d_dA, const cuDoubleComplex *d_buffer, const double *d_wbuff, int *dims);
__global__ void calcsstCUDA(cufftDoubleComplex *dA, const cufftDoubleComplex *A, double gamma2, double w0, cufftDoubleComplex *buffer, int dim2, int totaldim);

#endif