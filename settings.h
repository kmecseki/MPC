#pragma once
#ifndef MPC_SETTINGS_H
#define MPC_SETTINGS_H

#define USEGPU 1 // Only for low res runs, set to 0 for MPI
#define READWIS 1 // read existing wisdom for FFT

/* To use GPU for FFT as well */
#if USEGPU == 1
#include <cufft.h>
typedef cufftHandle fftw_plan;
#define fftw_destroy_plan cufftDestroy
#else
#include <fftw3.h>
typedef fftw_complex cufftDoubleComplex;
typedef fftw_complex cuDoubleComplex;
#endif

#endif
