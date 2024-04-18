#pragma once

#ifndef MPC_PROPAGATION_H
#define MPC_PROPAGATION_H
#include "settings.h"

void calcexpR(cufftDoubleComplex *expR, int *signmem, double z, double Cav, double k0, double dzz, double *wr, int *dims);
void fftshift(cufftDoubleComplex *in, int *dims, cufftDoubleComplex *buffer, int axis);
void calc1oes (double *BP, int *dims, double *r, double *vald1oes, double *y);
void ssfmprop(cufftDoubleComplex *A, int *dims, int sst_on, double dzz, double *betas, double alpha, int *signmem, double z, double Cav, double k0, double *wr, fftw_plan pR1fft, fftw_plan pR1ifft, fftw_plan pR2fft, fftw_plan pR2ifft, fftw_plan pTfft, fftw_plan pTifft, double *w, double gamma2, double w0, int plasm, double deltat, char *gas, double rho_c, double rho_nt, double n0, double *puls_e, double *r, double *BP1, double *BP2, double *y1, double *y2, double *bps1, double *bps2, double Ab_M, int change, int mpirank);
void linearstep(cufftDoubleComplex *A, cuDoubleComplex *exp_D0, cuDoubleComplex *expR, fftw_plan pTfft, fftw_plan pR2fft, fftw_plan pR1fft, int *dims);

void createPlans(int readin, int *dims, const char* cpath, fftw_plan *pR1fft, fftw_plan *pR1ifft, fftw_plan *pR2fft, fftw_plan *pR2ifft, fftw_plan *pTfft, fftw_plan *pTifft);
void setStep(int *dims, double *dzz, cufftDoubleComplex *C1, cufftDoubleComplex *C2, double def_err, double dzmin, double *y1, double *y2, int *bad, double *err);
void sendmpi(cufftDoubleComplex *A, MPI_Datatype datatype, int *dims, int rankto);
void recmpi(cufftDoubleComplex *A, MPI_Datatype datatype, int *dims, int rankfrom);

#endif