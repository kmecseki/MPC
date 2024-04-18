#include "kernels.h"

__global__ void memsetCUDA(double* array, size_t size) {
    
    /* memset for CUDA */
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = 0.0;
    }
}


__global__ void spatintegralCUDA(const cufftDoubleComplex *C, double *result, int *dims) {
    
    /* spatial integration */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dims[2]) {
        double sum = 0; // local sum for each k
        for (int i=0; i<dims[0]*dims[1]; i++) {
            sum += C[idx*dims[1]*dims[0]+i].x * C[idx*dims[1]*dims[0]+i].x + C[idx*dims[1]*dims[0]+i].y * C[idx*dims[1]*dims[0]+i].y;
        }
        result[idx] = sum;
    }
}


__device__ __forceinline__  double cexpCUDAr(cuDoubleComplex z) {
    
    /* cexp for CUDA */
    double real = cuCreal(z);
    double imag = cuCimag(z);
    double exp_real = exp(real);
    return exp_real * cos(imag);
}


__device__ __forceinline__  double cexpCUDAi(cuDoubleComplex z) {
    
    /* cexp for CUDA */
    double real = cuCreal(z);
    double imag = cuCimag(z);
    double exp_real = exp(real);
    return exp_real * sin(imag);
}


__global__ void calcexpD0CUDA(double dzz, double alpha, double *betas, cuDoubleComplex *exp_D0, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dims) {
        cuDoubleComplex z0 = make_cuDoubleComplex(-alpha/2.0 * dzz/2.0, betas[idx] * dzz/2.0);
        exp_D0[idx].x = cexpCUDAr(z0);
        exp_D0[idx].y = cexpCUDAi(z0);
    }
}


__global__ void propagTCUDA(cuDoubleComplex *A, const cuDoubleComplex *exponent, int *dims) {

    int k = blockIdx.x;
    int i = threadIdx.x;
    if (k < dims[2] && i < dims[0] * dims[1]) {
        A[k * dims[0] * dims[1] + i] = cuCmul(A[k * dims[0] * dims[1] + i], exponent[k]);
    }
}


__global__ void propagRCUDA(cuDoubleComplex *A, const cuDoubleComplex *exponent, int *dims, int dir) {
    
    int k = blockIdx.x;
    int j = threadIdx.y;
    int i = threadIdx.x;
    if (dir == 1) {
        if (k < dims[2] && j < dims[1] && i < dims[0]) {
            A[k * dims[0] * dims[1] + j * dims[0] + i] = cuCmul(A[k * dims[0] * dims[1] + j * dims[0] + i], cuConj(exponent[j]));
        }
    }
    else if (dir == 2) {
        if (k < dims[2] && j < dims[1] && i < dims[0]) {
            A[k * dims[0] * dims[1] + j * dims[0] + i] = cuCmul(A[k * dims[0] * dims[1] + j * dims[0] + i], cuConj(exponent[i]));
        }
    }
}


__global__ void calcdACUDA(cuDoubleComplex *dA, const cuDoubleComplex *buffer, const double *wbuff, int *dims) {
    
    int i = threadIdx.x;
    int k = blockIdx.x;

    if (k < dims[2] && i < dims[0] * dims[1]) {
        dA[k * dims[0] * dims[1] + i] = cuCmul(make_cuDoubleComplex(0.0, 1.0), buffer[k * dims[0] * dims[1] + i]);
        dA[k * dims[0] * dims[1] + i].x = dA[k * dims[0] * dims[1] + i].x * -1.0 * wbuff[k];
        dA[k * dims[0] * dims[1] + i].y = dA[k * dims[0] * dims[1] + i].y * -1.0 * wbuff[k];
    }
}


__global__ void calcsstCUDA(cufftDoubleComplex *dA, const cufftDoubleComplex *A, double gamma2, double w0, cufftDoubleComplex *buffer, int dim2, int totaldim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totaldim) {

        dA[idx].x = dA[idx].x / ((double)(dim2));
        dA[idx].y = dA[idx].y / ((double)(dim2));
        cufftDoubleComplex first = cuCmul(dA[idx], cuConj(A[idx]));
        first.x = first.x * 2.0;
        first.y = first.y * 2.0; 
        cufftDoubleComplex second = cuCmul(A[idx], cuConj(dA[idx]));

        buffer[idx].x = gamma2 / w0 * (first.x + second.x);
        buffer[idx].y = gamma2 / w0 * (first.y + second.y);
    }
}