#include "plasma.h"
#include "constants.h"

void ionizpot(const char* gas, double *I1, double *I2) {
    
    if (strcmp(gas,"Argon") == 0) {
        *I1 = 15.75962;
        *I2 = 27.62967;
    }

    if (strcmp(gas,"Helium") == 0) {
        *I1 = 24.58741;
        *I2 = 54.41778;
    }

    if (strcmp(gas,"Neon") == 0) {
        *I1 = 21.5646;
        *I2 = 40.96328;
    }

    if (strcmp(gas,"Air") == 0) { //% N2*mol/mol_air+O2*mol/mol_air+Ar*mol/mol_air
        *I1 = 0.78084 * 14.53414 + 0.20946 * 13.61806 + 0.00934 * 15.75962; 
        *I2 = 0.78084 * 29.6013 + 0.20946 * 35.11730 + 0.00934 * 27.62967;
    }

    if (strcmp(gas,"Nitrogen") == 0) {
        *I1 = 14.53414;
        *I2 = 29.6013;
    }

    if (strcmp(gas,"Xenon") == 0) {
        *I1 = 12.1298;
        *I2 = 21.20979;
    }
 
    if (strcmp(gas,"Krypton") == 0) {
        *I1 = 13.99961;
        *I2 = 24.35985;
    }
}


void ADK(double *ionizationrates1, double *ionizationrates2, int *dims, double *C_nl, double *f_lm, double *nq, double *mq, double *Ei, double *E) {
  
    double Kon1, Kon2, Kon3, Kon4, Kon5, Kon6;
    Kon1 = pow(abs(C_nl[0]),2.0) * sqrt(6.0/M_PI) * f_lm[0] * Ei[0];
    Kon2 = pow(abs(C_nl[1]),2.0) * sqrt(6.0/M_PI) * f_lm[1] * Ei[1];
    Kon3 = 2.0 * pow(2.0 * Ei[0], 3.0/2.0);
    Kon4 = 2.0 * pow(2.0 * Ei[1], 3.0/2.0);
    Kon5 = 2.0 * nq[0] - mq[0] - 3.0/2.0;
    Kon6 = 2.0 * nq[1] - mq[1] - 3.0/2.0;
           
    #pragma omp parallel for
    for (int i=0; i<dims[0] * dims[1] * dims[2]; i++) {
            ionizationrates1[i] = Kon1 * pow(Kon3 /E[i], Kon5) * exp(-Kon3 / (3.0 * E[i]));
            ionizationrates2[i] = Kon2 * pow(Kon4 /E[i], Kon6) * exp(-Kon4 / (3.0 * E[i]));
        }
}


void Natoms(const char *gas, double *Natm, double *sigm_coll) {  

    /* number density at 1bar in m^-3
    densities in kg/l
    Mx atomic weight g/mol */

    if (strcmp(gas,"Argon") == 0) {
        double MAr = 39.941;
        double densAr = 1.784 * 1e3;
        *Natm = densAr * NAVO / MAr; 
        *sigm_coll = 1.57e-20;
    }

    if (strcmp(gas,"Helium") == 0) {
        double MHe = 4.002602;
        double densHe = 0.166e3;
        *Natm = densHe * NAVO / MHe; 
        *sigm_coll = 6.11e-20;
    }

    if (strcmp(gas,"Neon") == 0) {
        double MNe = 20.1797;
        double densNe = 0.9002e3;
        *Natm = densNe * NAVO / MNe; 
        *sigm_coll = 1.65e-20;
    }

    if (strcmp(gas,"Air") == 0) {
        double Mair = 28.97;
        double densair = 1.205e3; //% at sea level
        *Natm = densair * NAVO / Mair; 
        *sigm_coll = 10e-20;
    }

    if (strcmp(gas,"Nitrogen") == 0) {
        double MN = 14.0067;
        double densN = 1.65e3;
        *Natm = densN * NAVO / MN; 
        *sigm_coll = 10.2e-20;
    }

    if (strcmp(gas,"Xenon")==0) {
        double MXe = 5.761;
        double densXe = 5.86e3;
        *Natm = densXe * NAVO / MXe; 
        *sigm_coll = 3.55e-20;
    }
 
    if (strcmp(gas,"Krypton") == 0) {
        double MKr = 83.798;
        double densKr = 3.749e3;
        *Natm = densKr * NAVO / MKr; 
        *sigm_coll = 1.15e-20;
    }
} 


void ioniz(cufftDoubleComplex *A, int *dims, double w0, double deltat, char* gas, double rho_c, double *w, double rho_nt, double n0, double *tau_c, double *freeelectrons_sp) {

    double *Ip;
    double Kon, k0;
    double *E0, *datpuls, *Iion, *f_lm, *nq, *lq, *mq, *C_nl;
    double I1, I2, ve;
    double *W_adk1, *W_adk2, *W_ava1, *Rateint;
    double Natm, sigm_coll;
    double *ions1, *ions2, *sigma_pla;
    Ip = (double *)malloc(dims[0] * dims[1] * dims[2] * sizeof(double));
    E0 = (double *)malloc(dims[0] * dims[1] * dims[2] * sizeof(double));
    datpuls = (double *)malloc(dims[0] * dims[1] * dims[2] * sizeof(double));
    W_adk1 = (double *)malloc(dims[0] * dims[1] * dims[2] * sizeof(double));
    W_adk2 = (double *)malloc(dims[0] * dims[1] * dims[2] * sizeof(double));
        
    #pragma omp parallel for private(i)
    for (int i=0; i<dims[2]*dims[1]*dims[0]; i++) {
        Ip[i] = pow(cuCabs(A[i]), 2) * HBAR * w0;
        if (Ip[i] == 0) {
            Ip[i] = 1e-30;
        }
    }
    
    Kon = deltat / T0;
    k0 = w0 / C0;
    
    /* Pulse */
    #pragma omp parallel for
    for (int i=0; i<dims[2]*dims[1]*dims[0]; i++) {
        E0[i] = sqrt(2.0 * ZW0 * Ip[i] / NC) * FCTR; // FCTR needs to be here due to the resolution
        datpuls[i] = fabs(E0[i]) / EF;   
    }
    
    /* Ionization rates */

    Iion = (double *)malloc(2 * sizeof(double));
    nq = (double *)malloc(2 * sizeof(double));
    f_lm = (double *)malloc(2 * sizeof(double));
    lq = (double *)malloc(2 * sizeof(double));
    mq = (double *)malloc(2 * sizeof(double));
    C_nl = (double *)malloc(2 * sizeof(double));
    ionizpot(gas, &I1, &I2);
    Iion[0] = I1 / IH;
    Iion[1] = I2 / IH;
    nq[0] = pow(2 * Iion[1],-0.5);
    nq[1] = 2.0 * pow(2 * Iion[2],-0.5);
    // f_lm for first two s subshells l = 0 and m = 0 (magnetic quantum number)
    //f_lm = ((2*lq+1).*factorial(lq+abs(mq)))./(2.^abs(mq).*factorial(abs(mq)).*factorial(lq-abs(mq)));
    f_lm[0] = 1.0;
    f_lm[1] = 1.0;
    lq[0] = 0.0;
    lq[1] = 0.0;
    mq[0] = 0.0;
    mq[1] = 0.0;
    
    C_nl[0] = sqrt(pow(2.0,2.0*nq[0]) * pow(nq[0] * tgamma(nq[0]+lq[0]+1.0) * tgamma(nq[0]-lq[0]),-1.0));
    C_nl[1] = sqrt(pow(2.0,nq[1]) * pow(nq[1] * tgamma(nq[1]+lq[1]+1.0) * tgamma(nq[1]-lq[1]),-1.0));
    ADK(W_adk1, W_adk2, dims, C_nl, f_lm, nq, mq, Iion, datpuls);
    free(Iion);
    free(datpuls);
    free(nq);
    free(f_lm);
    free(lq);
    free(mq);
    free(C_nl);
    
    #pragma omp parallel for
    for (int i=0; i<dims[2]*dims[1]*dims[0]; i++) {
        if (isnan(W_adk1[i])) {
            W_adk1[i] = 0.0;
        }
        if (isnan(W_adk2[i])) {
            W_adk2[i] = 0.0;
        }
    }
    W_ava1 = (double *)malloc(dims[0] * dims[1] * dims[2] * sizeof(double));
    sigma_pla = (double *)malloc(dims[0] * dims[1] * dims[2] * sizeof(double));    
    Natoms(gas, &Natm, &sigm_coll);
    double K1, K2, K3, K4, K5;
    
    K1 = 2.0 * pow(EL0,2.0);
    K2 = 4.0 * pow(ME,2.0);
    K3 = sigm_coll * rho_nt;
    K4 = k0 / (n0 * rho_c) * w0;
    K5 = pow(w0,2);
    #pragma omp parallel for collapse(2) private(ve)
    for (int k=0; k<dims[2]; k++) {
        for (int i=0; i<dims[0]*dims[1]; i++) {
            /* free electron velocity in E-field */
            ve = sqrt(K1 * pow(E0[k * dims[0] * dims[1] + i],2.0) /(K2 * pow(w[k] + w0, 2.0)));
            /* collision time or mean free time */
            tau_c[k*dims[0]*dims[1]+i] = 1.0 / (K3 * ve);
            sigma_pla[k*dims[0]*dims[1]+i] = K4 * tau_c[k * dims[0] * dims[1] + i] / (1.0 + K5 * pow(tau_c[k * dims[0] * dims[1] + i], 2.0));
            W_ava1[k*dims[0]*dims[1]+i] = sigma_pla[k * dims[0] * dims[1] + i] * Ip[k * dims[0] * dims[1] + i] / I1;
        }
    }
    
    free(Ip);
    free(E0);
    
    Rateint = (double *)calloc(dims[0] * dims[1], sizeof(double));
    ions1 = (double *)calloc((dims[0] * dims[1] * (dims[2] + 1)), sizeof(double));
    ions2 = (double *)calloc((dims[0] * dims[1] * (dims[2] + 1)), sizeof(double));
    
    //#pragma omp parallel for collapse(2) shared(Rateint) reduction(+:Rateint)
    for (int k=(int)dims[2]-1; k>=0; k--) {
        for (int i=0; i<dims[0]*dims[1]; i++) {
            Rateint[i] = Rateint[i] + W_adk1[k * dims[0] * dims[1] + i] * Kon;
            //#pragma omp barrier
            Rateint[i] = Rateint[i] + (1.0 - (Rateint[i])) * W_ava1[k * dims[0] * dims[1] + i] * Kon;
            ions1[(k+1)*dims[0]*dims[1]+i] = 1.0 - exp(-Rateint[i]) - ions2[k * dims[0] * dims[1] + i];
            ions2[(k+1)*dims[0]*dims[1]+i] = ions2[k * dims[0] * dims[1] + i] + (W_adk2[k*dims[0]*dims[1]+i]) * Kon * ions1[k*dims[0]*dims[1]+i];
        }
    }   

    free(Rateint);
    free(W_ava1);
    free(W_adk1);
    free(W_adk2);
    
    #pragma omp parallel for
    for (int i=0; i<dims[2]*dims[1]*dims[0]; i++) {
        freeelectrons_sp[i] = (ions1[i] * 1.0 + ions2[i] * 2.0) * sigma_pla[i];
    }
   
    free(sigma_pla);
    free(ions1);
    free(ions2);
}