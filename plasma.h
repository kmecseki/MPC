#pragma once

#ifndef MPC_PLASMA_H
#define MPC_PLASMA_H
#include "settings.h"

void ionizpot(const char* gas, double *I1, double *I2);
void ADK(double *ionizationrates1, double *ionizationrates2, int *dims, double *C_nl, double *f_lm, double *nq, double *mq, double *Ei, double *E);
void Natoms(const char *gas, double *Natm, double *sigm_coll);
void ioniz(cufftDoubleComplex *A, int *dims, double w0, double deltat, char* gas, double rho_c, double *w, double rho_nt, double n0, double *tau_c, double *freeelectrons);

#endif