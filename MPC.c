#define _USE_MATH_DEFINES
#include <math.h>
#include <complex.h>
#include<fftw3.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>
#include <time.h>

#define ZW0 377.0
#define NC 1.0  // refractive index (set to 1)
#define FCTR 0.89 // factor to account for no electric field oscillations 
#define IH 27.21
#define EF 5.14220642e11
#define ME 9.1094e-31
#define EL0 1.602e-19
#define NAVO 6.02214e+23
#define T0 2.4188843265e-17    
#define C0 300000000.0
#define HBAR 1.054560652926899e-34
#define PROC 12

////afs/slac/package/intel_tools/2015u2/bin/icc -o fullrun4 -O3 -lfftw3_threads -lfftw3 -lpthread -openmp source_code_name.c
/// current wisdom created on psanagpu111


void ionizpot(const char* gas, double *I1, double *I2);
void ADK(double *ionizationrates1, double *ionizationrates2, int *dims, double *C_nl, double *f_lm, double *nq, double *mq, double *Ei, double *E);
void Natoms(const char *gas, double *Natm, double *sigm_coll); 
void calcexpR(complex double *expR, int *signmem, double z, double Cav, double k0, double dzz, double *wr, int *dims);
void fftshift(complex double *in, int *dims, complex double *buffer, int axis);
void ioniz(complex double *A, int *dims, double w0, double deltat, char* gas, double rho_c, double *w, double rho_nt, double n0, double *tau_c, double *freeelectrons);
void calc1oes (double *BP, int *dims, double *r, double *vald1oes, double *y);
void ssfmprop(double complex *A, int *dims, int sst_on, double dzz, double *betas, double alpha, int *signmem, double z, double Cav, double k0, double *wr, fftw_plan pR1fft, fftw_plan pR1ifft, fftw_plan pR2fft, fftw_plan pR2ifft, fftw_plan pTfft, fftw_plan pTifft, double *w, double gamma2, double w0, int plasm, double deltat, char *gas, double rho_c, double rho_nt, double n0, double *puls_e, double *r, double *BP1, double *BP2, double *y1, double *y2, double *bps1, double *bps2, double Ab_M, int change);
void createPlans(int *dims, fftw_plan *pR1fft, fftw_plan *pR1ifft, fftw_plan *pR2fft, fftw_plan *pR2ifft, fftw_plan *pTfft, fftw_plan *pTifft);
void setStep(int *dims, double *dzz, double complex *C1, double complex *C2, double def_err, double dzmin, double *y1, double *y2, int *bad, double *err);

void main(int argc, char *argv[]) {
    
    //1. Read parameter file.
    FILE *fparp, *fmarp, *fmaip, *fbp1p, *fbp2p, *ftemp, *fspep, *fothp, *foutp;
    char fparn[80], fmarn[80], fmain[80], fbp1n[80], fbp2n[80], ftemn[80], fspen[80], fothn[80], foutn[80];
    //const char *cpath = "/tmp/tmp.A4EplIZ4xJ/"; 
    const char *cpath = "/reg/d/psdm/xpp/xpp12216/results/MCPdata/"; 
    //const char *cpath = "/reg/neh/home/kmecseki/broad3/3Dfortest/3D/files/data/"; 
    
    snprintf(fparn, 80, "%s/%s", cpath, "param.bin"); 
    snprintf(fmarn, 80, "%s/%s", cpath, "datar.bin");
    snprintf(fmain, 80, "%s/%s", cpath, "datai.bin");
    snprintf(fbp1n, 80, "%s/%s", cpath, "bp1.bin");
    snprintf(fbp2n, 80, "%s/%s", cpath, "bp2.bin");
    snprintf(ftemn, 80, "%s/%s", cpath, "temp.bin");
    snprintf(fspen, 80, "%s/%s", cpath, "spec.bin");
    snprintf(fothn, 80, "%s/%s", cpath, "other.bin");
    snprintf(foutn, 80, "%s/%s", cpath, "out.bin");
    
    fparp = fopen(fparn,"r");
    fmarp = fopen(fmarn,"r");
    fmaip = fopen(fmain,"r");
    fbp1p = fopen(fbp1n,"wb");
    fbp2p = fopen(fbp2n,"wb");
    ftemp = fopen(ftemn, "wb");
    fspep = fopen(fspen, "wb");
    fothp = fopen(fothn, "w");
    foutp = fopen(foutn, "wb");
      
    char gas[9];
    int i, k, dims[3], nstep;
    int signmem, sst_on, plasm;
    int change, bad;
    double complex *A, *C1, *C2;
    double *Ar, *Ai;
    double *betas, *wr, *w, *r, *temp, *spec;
    double dist, dzz, alpha, err, def_err, dzmin, dr;
    double z, Cav, k0;
    fftw_plan pR1fft, pR1ifft, pR2fft, pR2ifft, pTfft, pTifft;
    double gamma2, w0, deltat, rho_c, rho_nt, n0, puls_e, Ab_M;
    double bps1 = 0.0, bps2 = 0.0;
    double y1[4], y2[4];
    double *BP1, *BP2;
        
    fread(dims, sizeof(int),3,fparp);
    printf("Matrix dimensions are: %d x %d x %d. \n", dims[0],dims[1],dims[2]);
    
    printf("DEBUG: Allocating memory...\n");
    A = (complex double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(complex double));
    Ar = (double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(double));
    Ai = (double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(double));
    C1 = (complex double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(complex double));
    C2 = (complex double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(complex double));
    temp = (double *)malloc(dims[2] * sizeof(double));
    spec = (double *)malloc(dims[2] * sizeof(double));
    betas = (double *)malloc(dims[2] * sizeof(double));
    wr = (double *)malloc(dims[1] * sizeof(double));
    w = (double *)malloc(dims[2] * sizeof(double));
    r = (double *)malloc(dims[1] * sizeof(double));
    BP1 = (double *)malloc(dims[0] * sizeof(double));
    BP2 = (double *)malloc(dims[0] * sizeof(double));
    printf("\bDone!\n");
    
    printf("Now reading in data matrix...\n");
    fread(Ar, sizeof(double),dims[0]*dims[1]*dims[2],fmarp);
    fread(Ai, sizeof(double),dims[0]*dims[1]*dims[2],fmaip);
    fclose(fmaip);
    fclose(fmarp);
    #pragma omp parallel for
    for (i=0; i<dims[0]*dims[1]*dims[2]; i++) {
        A[i] = Ar[i] + I*Ai[i];
    }
    free(Ar);
    free(Ai);
    printf("\bDone!\n");
    
    printf("Now reading in parameters...\n");

    fread(&dist, sizeof(double),1,fparp);
    fread(&dzz, sizeof(double),1,fparp);
    fread(betas, sizeof(double),dims[2],fparp);
    fread(&alpha, sizeof(double),1,fparp);
    fread(&Cav, sizeof(double),1,fparp);
    fread(&k0, sizeof(double),1,fparp);
    fread(wr, sizeof(double),dims[1],fparp);
    fread(&sst_on, sizeof(int),1,fparp);
    fread(w, sizeof(double),dims[2],fparp);
    fread(&gamma2, sizeof(double),1,fparp);
    fread(&w0, sizeof(double),1,fparp);
    fread(&plasm, sizeof(int),1,fparp);
    fread(&deltat, sizeof(double),1,fparp);
    fread(&rho_c, sizeof(double),1,fparp);
    fread(&rho_nt, sizeof(double),1,fparp);
    fread(&n0, sizeof(double),1,fparp);
    fread(&puls_e, sizeof(double),1,fparp);
    fread(&Ab_M, sizeof(double),1,fparp);
    fread(r, sizeof(double),dims[1],fparp);
    fread(&dr, sizeof(double),1,fparp);
    fread(&def_err, sizeof(double),1,fparp);
    fread(&dzmin, sizeof(double),1,fparp);
    fread(gas, sizeof(char),9,fparp);
    
    
    fclose(fparp);
    printf("\bDone!\n");
    
    printf("Creating FFT plans...\n");
    fftw_init_threads();
    fftw_plan_with_nthreads(PROC);
    createPlans(dims, &pR1fft, &pR1ifft, &pR2fft, &pR2ifft, &pTfft, &pTifft);
    printf("\bDone!\n");
   
    
    err = 0;
    bad = 0;
    z = 0;
    signmem = 1;
    nstep = 0;
    
    while (z<dist) {
        
        #pragma omp parallel
        {
                memcpy(C1, A, dims[0]*dims[1]*dims[2]* sizeof(complex double));
                memcpy(C2, A, dims[0]*dims[1]*dims[2]* sizeof(complex double));
        }
        
        ssfmprop(C1, dims, sst_on, 2*dzz, betas, alpha, &signmem, z, Cav, k0, wr, pR1fft, pR1ifft, pR2fft, pR2ifft, pTfft, pTifft, w, gamma2, w0, plasm, deltat, gas, rho_c, rho_nt, n0, &puls_e, r, BP1, BP2, y1, y2, &bps1, &bps2, Ab_M, 0);
        ssfmprop(C2, dims, sst_on, dzz, betas, alpha, &signmem, z, Cav, k0, wr, pR1fft, pR1ifft, pR2fft, pR2ifft, pTfft, pTifft, w, gamma2, w0, plasm, deltat, gas, rho_c, rho_nt, n0, &puls_e, r, BP1, BP2, y1, y2, &bps1, &bps2, Ab_M, 0);
        ssfmprop(C2, dims, sst_on, dzz, betas, alpha, &signmem, z, Cav, k0, wr, pR1fft, pR1ifft, pR2fft, pR2ifft, pTfft, pTifft, w, gamma2, w0, plasm, deltat, gas, rho_c, rho_nt, n0, &puls_e, r, BP1, BP2, y1, y2, &bps1, &bps2, Ab_M, 1);

        setStep(dims, &dzz, C1, C2, def_err, dzmin, y1, y2, &bad, &err);

        if (dr>(y1[1]+y2[1])/2.0) {
            printf("ERROR Rdim too small, nonlinear effects pull the beam too small for this resolution.\n ");
            exit(0);
        }
        if (bad==1) {
           continue;
        }
        else {
            #pragma omp parallel for 
                for (i=0; i<dims[2]*dims[1]*dims[0]; i++) {
                    A[i] = (4.0/3.0)*C2[i] -(1.0/3.0)*C1[i];
                }
        }

        z = z+2.0*dzz;
        memset(temp, 0, dims[2] * sizeof(double));
      
        #pragma omp parallel sections
        {
        #pragma omp section
            {
            #pragma omp parallel for
            for (k=0; k<dims[2]; k++) {
                //#pragma omp parallel for private(i,k) shared(temp) reduction(+: temp)
                for (i=0; i<dims[1]*dims[0]; i++) {
                    temp[k] += A[k*dims[1]*dims[0]+i];
                }
                temp[k] = cabs(temp[k]);
            }
            }
        #pragma omp section
            {
            memcpy(C2, A, dims[0]*dims[1]*dims[2]* sizeof(complex double));
            memset(spec, 0, dims[2] * sizeof(double));
            fftw_execute_dft(pTfft, C2, C2);    
            //fftshift(C2, dims, C1, 3); % This was taking so long I took it out, we can do this in matlab.
            
            #pragma omp parallel for
            for (k=0; k<dims[2]; k++) {
                //#pragma omp parallel for private(i,k) shared(spec) reduction(+: spec)
                for (i=0; i<dims[1]*dims[0]; i++) {
                    spec[k] += C2[k*dims[1]*dims[0]+i];
                }
                spec[k] = cabs(spec[k]);
            }
            } 
        }
        #pragma omp parallel sections
        {
            #pragma omp section
                fwrite(BP1, sizeof(double), dims[1], fbp1p);
            #pragma omp section
                fwrite(BP2, sizeof(double), dims[1], fbp2p);
            #pragma omp section
                fwrite(temp, sizeof(double), dims[2], ftemp); 
            #pragma omp section
                fwrite(spec, sizeof(double), dims[2], fspep);
        }
        fprintf(fothp, "%f\t%f\n", err, dzz);
        if (z>dist) {
            z = dist;
        }
        
        printf("We are here: %5.2f", z*100.0/dist );
        nstep++;
        //exit(0);
        //if (nstep == 1)
        //{        
        //    z=dist; //csak egy kor
        //}
        //exit(0);
    }

fwrite(A, sizeof(double), 2*dims[0]*dims[1]*dims[2], foutp);
fclose(fbp1p);
fclose(fbp2p);
fclose(ftemp);
fclose(fspep);
fclose(fothp);
fclose(foutp);
        
fftw_destroy_plan(pR1fft);
fftw_destroy_plan(pR1ifft);
fftw_destroy_plan(pR2fft);
fftw_destroy_plan(pR2ifft);
fftw_destroy_plan(pTfft);
fftw_destroy_plan(pTifft);
fftw_cleanup(); 
fftw_cleanup_threads();

free(A);
free(C1);
free(C2);
free(temp);
free(spec);
free(betas);
free(wr);
free(w);
free(r);
free(BP1);
free(BP2);

}        
        
   
void ssfmprop(double complex *A, int *dims, int sst_on, double dzz, double *betas, double alpha, int *signmem, double z, double Cav, double k0, double *wr, fftw_plan pR1fft, fftw_plan pR1ifft, fftw_plan pR2fft, fftw_plan pR2ifft, fftw_plan pTfft, fftw_plan pTifft, double *w, double gamma2, double w0, int plasm, double deltat, char* gas, double rho_c, double rho_nt, double n0, double *puls_e, double *r, double *BP1, double *BP2, double *y1, double *y2, double *bps1, double *bps2, double Ab_M, int change) {   
   
    int i, j, k;
    int signum;
    double *freeelectrons_sp;
    double complex *exp_D0, *buffersmall, *expR, *buffer2, *buffer, *dA, *A_nl;
    double *tau_c;
    //printf("DEBUG: Allocating memory for ssfm...\n");
    exp_D0 = (double complex*)malloc(dims[2] * sizeof(double complex));
    expR = (double complex*)malloc(dims[0] * sizeof(double complex));
    buffersmall = (complex double *)malloc(dims[2] * sizeof(complex double));
    //buffer2 = (complex double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(complex double));
    buffer = (complex double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(complex double));
    dA = (complex double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(complex double));
    tau_c = (double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(double));
    freeelectrons_sp = (double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(double));
    A_nl = (complex double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(complex double));
    //printf("\bDone!\n");
    
    // Linear step - half step
   // printf("Linear step...\n");

    #pragma omp parallel for
    for (i=0; i<dims[2]; i++) {
        //exp_D0[i] = exp(creal(dzz/2.0*(I*betas[i]-alpha/2.0)))*(cos(cimag(dzz/2.0*(I*betas[i]-alpha/2.0)))+I*sin(cimag(dzz/2.0*(I*betas[i]-alpha/2.0))));//
        exp_D0[i] = cexp(dzz/2.0*(I*betas[i]-alpha/2.0));
    }
   
    fftshift(exp_D0, dims, buffersmall, 0);
    calcexpR(expR, signmem, z, Cav, k0, dzz/2, wr, dims); //a fenti ezt helyettesiti csak parallel

    //exit(0);
    //FILE *this1;
    //this1 = fopen("/tmp/tmp.A4EplIZ4xJ/expr.bin","wb");
    //fwrite(expR, sizeof(double), 2*dims[0], this1);
    //fclose(this1);
    //exit(0);
    fftw_execute_dft(pTfft, A, A);

    #pragma omp parallel for collapse(2)
        for (k=0; k<dims[2]; k++) {
            for (i=0; i<dims[0]*dims[1]; i++) {
                A[k*dims[0]*dims[1]+i] = A[k*dims[0]*dims[1]+i] * exp_D0[k];// * (double) dims[2]; // Removed conj!
            }
        }

    fftw_execute_dft(pTifft, A, A);
             
    fftw_execute_dft(pR2fft, A, A);
    
    #pragma omp simd collapse(3) 
    for (k=0; k<dims[2]; k++) {
        for (j=0; j<dims[1]; j++) {
            for (i=0; i<dims[0]; i++) {
                A[k*dims[0]*dims[1]+j*dims[1]+i] = A[k*dims[0]*dims[1]+j*dims[1]+i] * conj(expR[j]);
            }
        }
    }  
    
    fftw_execute_dft(pR2ifft, A, A);
    
    fftw_execute_dft(pR1fft, A, A);
    
    #pragma omp simd collapse(3) 
    for (k=0; k<dims[2]; k++) {
        for (j=0; j<dims[1]; j++) {
            for (i=0; i<dims[0]; i++) {
                A[k*dims[0]*dims[1]+j*dims[1]+i] = A[k*dims[0]*dims[1]+j*dims[1]+i] * conj(expR[i]);
            }
        }
    }
    
    fftw_execute_dft(pR1ifft, A, A);
    
    #pragma omp parallel for  
    for (k=0; k<dims[2]*dims[1]*dims[0]; k++) {
        A[k] = A[k] / ((double) (dims[2]*dims[1]*dims[0]));
    }

    // Nonlinear step - full step 
    // printf("Nonlinear step...\n");
       
   //            FILE *this3;
   // this3 = fopen("/tmp/tmp.fSt7RsB2I8/kozben.bin","wb");
    //fwrite(A, sizeof(double), 2*dims[2]*dims[1]*dims[0], this3);
   /// fclose(this3);
    
    if (sst_on == 1) {
    #pragma omp parallel for
       for (i=0; i<dims[0]*dims[1]*dims[2]; i++) {
           buffer[i] = A[i];
       }
    
   // FILE *this3;
   // this3 = fopen("/reg/d/psdm/xpp/xpp12216/results/MCPdata/withfftshift.bin","wb");
   // fwrite(dA, sizeof(double), 2*dims[2]*dims[1]*dims[0], this3);
   // fclose(this3);
     
    double *wbuff;
    wbuff = (double *)malloc(dims[2] * sizeof(double));
    memcpy (wbuff, w, dims[2] * sizeof(double) );
    #pragma omp parallel for
        for (k=0; k<dims[2]/2; k++) {
            w[k] = wbuff[dims[2]/2+k];
            w[dims[2]/2+k] = wbuff[k];
        }
    free(wbuff);
    fftw_execute_dft(pTfft, buffer, buffer);
    #pragma omp parallel for collapse(2)
    for (k=0; k<dims[2]; k++) {
           for (i=0; i<dims[0]*dims[1]; i++) {
               dA[k*dims[0]*dims[1]+i] = -1.0*I * buffer[k*dims[0]*dims[1]+i] * w[k]; //conj v no conj for w???
           }
    }

    fftw_execute_dft(pTifft, dA, dA);

       
  //  FILE *this4;
   // this4 = fopen("/reg/d/psdm/xpp/xpp12216/results/MCPdata/withoutfftshift.bin","wb");
   // fwrite(dA, sizeof(double), 2*dims[2]*dims[1]*dims[0], this4);
  //  fclose(this4);
    
  //  free(wbuff);
   // exit(0);

       #pragma omp parallel for
       for (k=0; k<dims[2]*dims[1]*dims[0]; k++) {
           dA[k] = dA[k] / ((double) (dims[2]));
       }

       #pragma omp parallel for  
       for (i=0; i<dims[0]*dims[1]*dims[2]; i++) {
           buffer[i] = gamma2/w0*(2.0*dA[i]*conj(A[i])+A[i]*conj(dA[i])); //Using buffer for a new thing, different than before, to save memory
       }
       
   }
   else {
       memset(buffer, 0, dims[0]*dims[1]*dims[2] * sizeof(complex double));
   }
   
   if (plasm == 1) {
       ioniz(A, dims, w0, deltat, gas, rho_c, w, rho_nt, n0, tau_c, freeelectrons_sp);
      #pragma omp parallel for
       for (i=0; i<dims[0]*dims[1]*dims[2]; i++) { 
           A_nl[i] = I * gamma2 * pow(cabs(A[i]),2.0)-buffer[i] - (1.0 + I * w0 * tau_c[i]) * rho_nt * freeelectrons_sp[i]/2.0;
       }
  
   }
   else {
       #pragma omp parallel for  
       for (i=0; i<dims[0]*dims[1]*dims[2]; i++) {
           A_nl[i] = I * gamma2 * pow(cabs(A[i]),2.0) - buffer[i];
       }
       
   }

  #pragma omp parallel for
   for (i=0; i<dims[0]*dims[1]*dims[2]; i++) {
       //A[i] = A[i] * cexp(dzz * A_nl[i]); 
       A[i] = A[i] * exp(creal(dzz * A_nl[i]))*(cos(cimag(dzz * A_nl[i]))+I*sin(cimag(dzz * A_nl[i])));
   }

    // Linear step - half step
  // printf("Linear step...\n");
   fftw_execute_dft(pTfft, A, A);
 #pragma omp parallel for collapse(2)
   for (k=0; k<dims[2]; k++) {
       for (i=0; i<dims[0]*dims[1]; i++) {
           A[k*dims[0]*dims[1]+i] = A[k*dims[0]*dims[1]+i] * exp_D0[k];// * (double) dims[2]; // Removed conj!
       }
   }
   
   fftw_execute_dft(pTifft, A, A);

   fftw_execute_dft(pR2fft, A, A);

   #pragma omp simd collapse(3)
   for (k=0; k<dims[2]; k++) {
      for (j=0; j<dims[1]; j++) {
            for (i=0; i<dims[0]; i++) {
                A[k*dims[0]*dims[1]+j*dims[1]+i] = A[k*dims[0]*dims[1]+j*dims[1]+i] * conj(expR[j]);
            }
       }
   }
       
   fftw_execute_dft(pR2ifft, A, A);
   
   fftw_execute_dft(pR1fft, A, A);

  #pragma omp simd collapse(3)
   for (k=0; k<dims[2]; k++) {
       for (j=0; j<dims[1]; j++) {
           for (i=0; i<dims[0]; i++) {
                A[k*dims[0]*dims[1]+j*dims[1]+i] = A[k*dims[0]*dims[1]+j*dims[1]+i] * conj(expR[i]);
            }
       }
   }
      
   fftw_execute_dft(pR1ifft, A, A);
     
  #pragma omp parallel for
  for (k=0; k<dims[2]*dims[1]*dims[0]; k++) {
      A[k] = A[k] / ((double) (dims[2]*dims[1]*dims[0]));
  }
   //   FILE *this3;
   //this3 = fopen("/tmp/tmp.fSt7RsB2I8/kozben.bin","wb");
   //fwrite(A, sizeof(double), 2*dims[2]*dims[1]*dims[0], this3);
   //fclose(this3);
   //exit(0);
   
  //memcpy(temp, A, dims[2]*dims[1]*dims[0]* sizeof(complex double)); 
   //        FILE *this2;
    //this2 = fopen("/tmp/tmp.fSt7RsB2I8/utana.bin","wb");
    //fwrite(A, sizeof(double), 2*dims[2]*dims[1]*dims[0], this2);
    //fclose(this2);
    //exit(0);
    //printf("Exitingggggg");
   
   if (change==1) {
       signum = ((int) (floor(z/Cav)+1) % 2) * 2-1;
       if (*signmem == -signum) {
           *puls_e = *puls_e*Ab_M;
           //printf("Most vesuznk el egy kis energiat ahahahha\n");
           *signmem *= -1;
       }
      // printf("This is now signum: %d and signmem: %d\n",signum, *signmem);
      // printf("z= %f, Cav = %f,floor(z/Cav) = %f, floor(z/Cav)+1  2= %d \n",z,Cav,floor(z/Cav),((int)(floor(z/Cav)+1) % 2));
   }
   
   //mexPrintf("Ide eljutottunk4\n");
   if (change==1) {
       memset(BP1, 0, dims[2] * sizeof(double));
       memset(BP2, 0, dims[2] * sizeof(double));
       //#pragma omp parallel for collapse(3) shared(BP1,BP2) reduction(+:BP1,BP2)
       for (i=0; i<dims[2]; i++) {
           for (j=0; j<dims[1]; j++) {
               for (k=0; k<dims[0]; k++) {
                   BP1[k] += pow(cabs(A[k+dims[0]*j+i*dims[0]*dims[1]]),2.0);
                   BP2[j] += pow(cabs(A[k+dims[0]*j+i*dims[0]*dims[1]]),2.0);
               }
           }
       }
       #pragma omp parallel sections
       {
           #pragma omp section
               calc1oes(BP1, dims, r, bps1, y1);
           #pragma omp section
               calc1oes(BP2, dims, r, bps2, y2);
       }
   }
   
   // FILE *this1;
   // this1 = fopen("/reg/d/psdm/xpp/xpp12216/results/MCPdata/ion.bin","wb");
   // fwrite(A, sizeof(double), 2*dims[0]*dims[1]*dims[2], this1);
   // fclose(this1);
    
   // exit(0);

   //printf("SSFM Done!\n");
   free(tau_c);
   free(freeelectrons_sp);
   free(A_nl);
   free(buffer);
   free(dA);
   free(exp_D0);
   free(expR);
   free(buffersmall);
    
}      

void ionizpot(const char* gas, double *I1, double *I2) {
    
    if (strcmp(gas,"Argon")==0) {
        *I1 = 15.75962;
        *I2 = 27.62967;
    }

    if (strcmp(gas,"Helium")==0) {
        *I1 = 24.58741;
        *I2 = 54.41778;
    }

    if (strcmp(gas,"Neon")==0) {
        *I1 = 21.5646;
        *I2 = 40.96328;
    }

    if (strcmp(gas,"Air")==0) { //% N2*mol/mol_air+O2*mol/mol_air+Ar*mol/mol_air
        *I1 = 0.78084*14.53414+0.20946*13.61806+0.00934*15.75962; 
        *I2 = 0.78084*29.6013+0.20946*35.11730+0.00934*27.62967;
    }

    if (strcmp(gas,"Nitrogen")==0) {
        *I1 = 14.53414;
        *I2 = 29.6013;
    }

    if (strcmp(gas,"Xenon")==0) {
        *I1 = 12.1298;
        *I2 = 21.20979;
    }
 
    if (strcmp(gas,"Krypton")==0) {
        *I1 = 13.99961;
        *I2 = 24.35985;
    }
}


void ADK(double *ionizationrates1, double *ionizationrates2, int *dims, double *C_nl, double *f_lm, double *nq, double *mq, double *Ei, double *E) {
    int i;  
    #pragma omp parallel for
    for (i=0; i<dims[0]*dims[1]*dims[2]; i++) {
            ionizationrates1[i] = pow(cabs(C_nl[0]),2.0) * sqrt(6.0/M_PI) * f_lm[0] * Ei[0] * pow(2.0 * pow(2.0*Ei[0],3.0/2.0) /E[i],2.0*nq[0]-mq[0]-3.0/2.0)*exp(-2.0*pow(2*Ei[0],3.0/2.0)/(3.0*E[i]));
            ionizationrates2[i] = pow(cabs(C_nl[1]),2.0) * sqrt(6.0/M_PI) * f_lm[1] * Ei[1] * pow(2.0 * pow(2.0*Ei[1],3.0/2.0) /E[i],2.0*nq[1]-mq[1]-3.0/2.0)*exp(-2.0*pow(2*Ei[1],3.0/2.0)/(3.0*E[i]));
        }
}
        
void Natoms(const char *gas, double *Natm, double *sigm_coll) {  

/* number density at 1bar in m^-3
 densities in kg/l
 Mx atomic weight g/mol */

if (strcmp(gas,"Argon")==0) {
    double MAr = 39.941;
    double densAr = 1.784*1e3;
    *Natm = densAr*NAVO/MAr; 
    *sigm_coll = 1.57e-20;
}

if (strcmp(gas,"Helium")==0) {
    double MHe = 4.002602;
    double densHe = 0.166e3;
    *Natm = densHe*NAVO/MHe; 
    *sigm_coll = 6.11e-20;
}

if (strcmp(gas,"Neon")==0) {
    double MNe = 20.1797;
    double densNe = 0.9002e3;
    *Natm = densNe*NAVO/MNe; 
    *sigm_coll = 1.65e-20;
}

if (strcmp(gas,"Air")==0) {
    double Mair = 28.97;
    double densair = 1.205e3; //% at sea level
    *Natm = densair*NAVO/Mair; 
    *sigm_coll = 10e-20;
}

if (strcmp(gas,"Nitrogen")==0) {
    double MN = 14.0067;
    double densN = 1.65e3;
    *Natm = densN*NAVO/MN; 
    *sigm_coll = 10.2e-20;
}

if (strcmp(gas,"Xenon")==0) {
    double MXe = 5.761;
    double densXe = 5.86e3;
    *Natm = densXe*NAVO/MXe; 
    *sigm_coll = 3.55e-20;
}
 
if (strcmp(gas,"Krypton")==0) {
    double MKr = 83.798;
    double densKr = 3.749e3;
    *Natm = densKr*NAVO/MKr; 
    *sigm_coll = 1.15e-20;
}

} 


void ioniz(complex double *A, int *dims, double w0, double deltat, char* gas, double rho_c, double *w, double rho_nt, double n0, double *tau_c, double *freeelectrons_sp) {

    int i, k;
    double *Ip;
    double Kon, k0;
    double *E0, *datpuls, *Iion, *f_lm, *nq, *lq, *mq, *C_nl;
    double I1, I2, ve;
    double *W_adk1, *W_adk2, *W_ava1, *Rateint;
    double Natm, sigm_coll;
    double *rate, *ions1, *ions2, *sigma_pla;
            
    Ip = (double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(double));
    E0 = (double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(double));
    datpuls = (double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(double));
    W_adk1 = (double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(double));
    W_adk2 = (double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(double));
        
    #pragma omp parallel for private(i)
    for (i=0; i<dims[2]*dims[1]*dims[0]; i++) {
        Ip[i] = pow(cabs(A[i]),2)*HBAR*w0;
        if (Ip[i]==0) {
            Ip[i] = 1e-30;
            //printf("IPP = %i",i);
        }
    }
    
    Kon = deltat/T0;
    k0 = w0/C0;
    
    /* Pulse */
    #pragma omp parallel for
    for (i=0; i<dims[2]*dims[1]*dims[0]; i++) {
        E0[i] = sqrt(2.0*ZW0*Ip[i]/NC)*FCTR; // FCTR needs to be here due to the resolution
        datpuls[i] = fabs(E0[i])/EF;   
    }
    
    /* Ionization rates */

    Iion = (double *)malloc(2 * sizeof(double));
    nq = (double *)malloc(2 * sizeof(double));
    f_lm = (double *)malloc(2 * sizeof(double));
    lq = (double *)malloc(2 * sizeof(double));
    mq = (double *)malloc(2 * sizeof(double));
    C_nl = (double *)malloc(2 * sizeof(double));
    
    ionizpot(gas, &I1, &I2);
    Iion[0] = I1/IH;
    Iion[1] = I2/IH;
    nq[0] = pow(2*Iion[1],-0.5);
    nq[1] = 2.0*pow(2*Iion[2],-0.5);
    // f_lm for first two s subshells l=0 and m=0 (magnetic quantum number)
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
    for (i=0; i<dims[2]*dims[1]*dims[0]; i++) {
        if isnan(W_adk1[i]) {
            W_adk1[i] = 0.0;
        }
        if isnan(W_adk2[i]) {
            W_adk2[i] = 0.0;
        }
    }
    W_ava1 = (double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(double));
    sigma_pla = (double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(double));    
    Natoms(gas, &Natm, &sigm_coll);
    #pragma omp parallel for collapse(2) private(ve)
    for (k=0; k<dims[2]; k++) {
       for (i=0; i<dims[0]*dims[1]; i++) {
            ve = sqrt(2.0 * pow(EL0,2.0) * pow(E0[k*dims[0]*dims[1]+i],2.0) /(4.0*pow(ME,2.0) * pow(w[k]+w0,2.0))); // free electron velocity in E-field 
            tau_c[k*dims[0]*dims[1]+i] = 1.0/(sigm_coll*rho_nt*ve); //% collision time or mean free time  
            sigma_pla[k*dims[0]*dims[1]+i] = k0/(n0*rho_c)*w0*tau_c[k*dims[0]*dims[1]+i]/(1.0+pow(w0,2)*pow(tau_c[k*dims[0]*dims[1]+i],2.0));
            W_ava1[k*dims[0]*dims[1]+i] = sigma_pla[k*dims[0]*dims[1]+i]*Ip[k*dims[0]*dims[1]+i]/I1;
       }
    }
    
    free(Ip);
    free(E0);
    
    //for (int i=0; i<dims[1]*dims[0]; i++) {
    //    Rateint[i] = 0.0;
    //}
    
    Rateint = (double *)calloc(dims[0]*dims[1], sizeof(double));
    ions1 = (double *)calloc((dims[0]*dims[1]*(dims[2]+1)), sizeof(double));
    ions2 = (double *)calloc((dims[0]*dims[1]*(dims[2]+1)), sizeof(double));
    // CHECK IF THE FOLLOWING IS CORRECT (RACE CONDITOIN MIGHT BE A PROBLEM)
    //#pragma omp parallel for collapse(2) shared(Rateint) reduction(+:Rateint)
    for (k=(int)dims[2]-1; k>=0; k--) {
        for (i=0; i<dims[0]*dims[1]; i++) {
            Rateint[i] = Rateint[i] + W_adk1[k*dims[0]*dims[1]+i]*Kon;
            //#pragma omp barrier
            Rateint[i] = Rateint[i] + (1.0- (Rateint[i]))*W_ava1[k*dims[0]*dims[1]+i]*Kon;
           // rate[k*dims[0]*dims[1]+i] = exp(-Rateint[i]); // Removed rate completely
            ions1[(k+1)*dims[0]*dims[1]+i] = 1.0-exp(-Rateint[i])-ions2[k*dims[0]*dims[1]+i];
            ions2[(k+1)*dims[0]*dims[1]+i] = ions2[k*dims[0]*dims[1]+i]+ (W_adk2[k*dims[0]*dims[1]+i])*Kon*ions1[k*dims[0]*dims[1]+i];
        }
    }
    
    free(Rateint);
    free(W_ava1);
    free(W_adk1);
    free(W_adk2);
    
    #pragma omp parallel for
    for (i=0; i<dims[2]*dims[1]*dims[0]; i++) {
        freeelectrons_sp[i] = (ions1[i]*1.0+ions2[i]*2.0)*sigma_pla[i];
    }
   
    free(sigma_pla);
    free(ions1);
    free(ions2);
}

void calcexpR(complex double *expR, int *signmem, double z, double Cav, double k0, double dzz, double *wr, int *dims) {
    int signum, i;
    signum = ((int)(floor(z/Cav)+1) % 2)*2-1;   
    double K;
    //%signum1 = mod(floor(z/Cav/2)+1,2)*2-1; % add cyl later
    //%signum2 = mod(floor(z/Cav/2)+2,2)*2-1;
    if (-*signmem == signum) {
        *signmem = -*signmem;
    }

    K = signum*(-I/(2.0*k0)*dzz*(-1.0));

    #pragma omp parallel for  
    for (i=0; i<dims[0]; i++) {
        expR[i] = cexp(signum*(-I/(2.0*k0)*dzz*(-1.0))*pow(wr[i]-(wr[dims[1]/2-1]+wr[dims[1]/2])/2.0,2));
    }

   // FILE *this;
   // this = fopen("/tmp/tmp.fSt7RsB2I8/miez.bin","wb");
   // fwrite(expR, sizeof(double), 2*dims[2], this);
   // fclose(this);
   // exit(0);
    //printf("Exitingggggg");
}

void fftshift(complex double *in, int *dims, complex double *buffer, int axis) {
    int i, j, k;

    if (axis==0) { //Special case for exp_D0
        memcpy ( buffer, in, dims[2] * sizeof(complex double) );
        #pragma omp parallel for
        for (i=0; i<dims[2]/2; i++) {
            in[i] = buffer[dims[2]/2+i];
            in[dims[2]/2+i] = buffer[i];
        }
    }            
    else {
        memcpy ( buffer, in, dims[0]*dims[1]*dims[2] * sizeof(complex double) );
        #pragma omp parallel num_threads(PROC)   
        {
        if (axis==1) {
            #pragma omp parallel for collapse(3)
            for (k=0; k<dims[2]; k++) {
                for (j=0; j<dims[1]; j++) {
                    for (i=0; i<(dims[0]/2); i++) {
                        #pragma omp parallel sections
                        {
                        #pragma omp section
                            in[k*dims[0]*dims[1]+j*dims[0]+i] = buffer[k*dims[0]*dims[1]+j*dims[0]+dims[0]/2+i];
                        #pragma omp section
                            in[k*dims[0]*dims[1]+j*dims[0]+dims[0]/2+i] = buffer[k*dims[0]*dims[1]+j*dims[0]+i];
                        }
                    }
                }
            }
        }
        if (axis==2) {
            #pragma omp parallel for collapse(3)
            for (k=0; k<dims[2]; k++) {
                for (j=0; j<dims[1]/2; j++) {
                    for (i=0; i<(dims[0]); i++) {
                        #pragma omp parallel sections
                        {
                        #pragma omp section
                            in[k*dims[0]*dims[1]+j*dims[0]+i] = buffer[k*dims[0]*dims[1]+(j+dims[1]/2)*dims[0]+i];
                        #pragma omp section
                            in[k*dims[0]*dims[1]+(j+dims[1]/2)*dims[0]+i] = buffer[k*dims[0]*dims[1]+j*dims[0]+i];
                        }
                    }
                }
            }
        }
        if (axis==3) {
            #pragma omp parallel for collapse(3)
            for (k=0; k<dims[2]/2; k++) {
                for (j=0; j<dims[1]; j++) {
                    for (i=0; i<(dims[0]); i++) {
                        #pragma omp parallel sections
                        {
                        #pragma omp section
                            in[k*dims[0]*dims[1]+j*dims[0]+i] = buffer[(k+dims[2]/2)*dims[0]*dims[1]+j*dims[0]+i];
                        #pragma omp section
                            in[(k+dims[2]/2)*dims[0]*dims[1]+j*dims[0]+i] = buffer[k*dims[0]*dims[1]+j*dims[0]+i];
                        }
                    }
                }
            }
        }
        }
    }
}

void calc1oes (double *BP, int *dims, double *r, double *vald1oes, double *y) {
    double bpex = 0;
    int posi1 = 0, posi2 = 0;
    double *normBP, halfBP;
    normBP = (double *)malloc(dims[0] * sizeof(double));
    //halfBP = (double *)malloc(dims[0]/2 * sizeof(double));
    memcpy(normBP, BP, dims[0]*sizeof(double));
    int k;
    #pragma omp parallel for
    for (k=0; k<dims[0]; k++) {   
       if (BP[k]>bpex)
           bpex = BP[k]; 
    }
   #pragma omp parallel for    
   for (k=0; k<dims[0]; k++) { 
       normBP[k] = BP[k]/bpex;
   }
   #pragma omp parallel for
   for (k=0; k<dims[0]/2; k++) {
       halfBP = fabs(normBP[k]-1.0/pow(exp(1.0),2.0));
       if (k==0)
           bpex = halfBP;
       else if (halfBP<bpex) {
           bpex = halfBP;
           posi1 = k;
       }
   }
   #pragma omp parallel for 
   for (k=dims[0]/2; k<dims[0]; k++) {
       halfBP = fabs(normBP[k]-1.0/pow(exp(1.0),2.0));
       if (k==dims[0]/2)
           bpex = halfBP;
       else if (halfBP<bpex) {
           bpex = halfBP;
           posi2 = k;
       }
   }
    
   *vald1oes = -r[posi1]+r[posi2];
   y[0] = r[posi1];
   y[1] = r[posi2];
   y[2] = BP[posi1];
   y[3] = BP[posi2];
           
}

void createPlans(int *dims, fftw_plan *pR1fft, fftw_plan *pR1ifft, fftw_plan *pR2fft, fftw_plan *pR2ifft, fftw_plan *pTfft, fftw_plan *pTifft) {
    
   complex double *buffer;
   int howmany, istride, idist;
   int *inembed;
   int *length;
   fftw_iodim64 *dim=malloc(1*sizeof(fftw_iodim64));
       if(dim==NULL){fprintf(stderr,"malloc failed\n");exit(1);}
   fftw_iodim64 *howmany_dims=malloc(2*sizeof(fftw_iodim64));
       if(howmany_dims==NULL){fprintf(stderr,"malloc failed\n");exit(1);}
   int howmany_rank;
   //int readnotwrite;
   
   //Reading in wisdom - comment out for new wisdom and reinstate the end of this function
   /*
   if (dims[0]==512) {
       const char* bop = "/reg/d/psdm/xpp/xpp12216/results/MCPdata/wisdom512";
       printf("Reading 512 wisdom.\n");
       readnotwrite = 0;
       fftw_import_wisdom_from_filename(bop);
   }
   else if (dims[0]==1024) {
       const char* bop = "/reg/d/psdm/xpp/xpp12216/results/MCPdata/wisdom1024";
       printf("Reading 1024 wisdom.\n");
       readnotwrite = 0;
       fftw_import_wisdom_from_filename(bop);
   }
   else {
       readnotwrite = 1;
       printf("Wisdom does not exist, creating one.\n");
   }
    */
   const char* bop = "/reg/d/psdm/xpp/xpp12216/results/MCPdata/wisdom512";
   fftw_import_wisdom_from_filename(bop);
   
   
   
   buffer = (complex double *)malloc(dims[0]*dims[1]*dims[2] * sizeof(complex double));
   
   dim[0].n = dims[0]; // An array of size rank, so always 1 in our case. This one is along dim0
   dim[0].is = 1;
   dim[0].os = 1;
   howmany_rank = 2;
   howmany_dims[0].n = dims[1];
   howmany_dims[0].is = dims[0];
   howmany_dims[0].os=  dims[0];
   howmany_dims[1].n = dims[2];
   howmany_dims[1].is =  dims[1]*dims[0];
   howmany_dims[1].os=  dims[1]*dims[0];
        
   *pR1fft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_FORWARD, FFTW_MEASURE);
   *pR1ifft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_BACKWARD, FFTW_MEASURE);
   //*pR1fft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_FORWARD, FFTW_PATIENT);
   //*pR1ifft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_BACKWARD, FFTW_PATIENT);
   
   dim[0].n = dims[1];
   dim[0].is =  dims[0];
   dim[0].os =  dims[0];
   howmany_rank = 2;
   howmany_dims[0].n = dims[0];
   howmany_dims[0].is = 1;
   howmany_dims[0].os= 1;
   howmany_dims[1].n = dims[2];
   howmany_dims[1].is =  dims[1]*dims[0];
   howmany_dims[1].os=  dims[1]*dims[0];
   
   *pR2fft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_FORWARD, FFTW_MEASURE);
   *pR2ifft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_BACKWARD, FFTW_MEASURE);
   //*pR2fft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_FORWARD, FFTW_PATIENT);
   //*pR2ifft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_BACKWARD, FFTW_PATIENT);
   
   dim[0].n = dims[2];
   dim[0].is =  dims[0]*dims[1];
   dim[0].os =  dims[0]*dims[1];
   howmany_rank = 2;
   howmany_dims[0].n = dims[0];
   howmany_dims[0].is = 1;
   howmany_dims[0].os= 1;
   howmany_dims[1].n = dims[1];
   howmany_dims[1].is =  dims[0];
   howmany_dims[1].os=  dims[0];
   //memcpy(buffer, A, dims[2]*dims[1]*dims[0]* sizeof(complex double));
   //memcpy(temp, expR, dims[0]* sizeof(complex double));
   *pTfft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_FORWARD, FFTW_MEASURE);
   *pTifft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_BACKWARD, FFTW_MEASURE);
   //*pTfft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_FORWARD, FFTW_PATIENT);
   //*pTifft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_BACKWARD, FFTW_PATIENT);
   //pTfft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, A, A, FFTW_FORWARD, FFTW_ESTIMATE);
   //pTifft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, A, A, FFTW_BACKWARD, FFTW_ESTIMATE);
   
   //Write out wisdom, uncomment for new hardware.
   //if (readnotwrite==1)
   //const char* bop = "/reg/d/psdm/xpp/xpp12216/results/MCPdata/wisdom1024";
   //fftw_export_wisdom_to_filename(bop);
   //exit(0);
   //
   free(buffer);
   free(dim);
   free(howmany_dims);
}

void setStep(int *dims, double *dzz, double complex *C1, double complex *C2, double def_err, double dzmin, double *y1, double *y2, int *bad, double *err) {
    int i,j;
    double sum1, sum2, err_fact, ujdz;
    //complex double *mC1, *mC2;
    
   // mC1 = (complex double *)malloc(dims[0]*dims[1] * sizeof(complex double));
   // mC2 = (complex double *)malloc(dims[0]*dims[1] * sizeof(complex double));
    
    *bad = 0;
    sum1 = 0;
    sum2 = 0;


    //#pragma omp parallel for default(shared) reduction(+:sum1, sum2)  //try private i?
    for (i = 0; i<dims[0]*dims[1]*dims[2]; i++) { 
        sum1 += pow(cabs(C2[i]-C1[i]), 2); 
        sum2 += pow(cabs(C2[i]), 2); 
        } 
    
    *err = sqrt(sum1)/sqrt(sum2);
    //printf("Sum1 = %f, Sum2 = %f\n ", sum1, sum2);
    printf("With step size: %.4f, the local error is: %.8g\nThe defined error is:%.8g\n ", *dzz, *err, def_err);

    err_fact = pow(2.0,(1.0/3.0)); //Split-step error
    if (*err>2.0*def_err) {
        // Decrease step size and calc new solution
        ujdz = *dzz/2.0;
        if (ujdz>dzmin) {
            *dzz = ujdz;
            printf("Step decreased, re-calculating...\n");
            *bad = 1;
            return;
        }
        else {
            // Must accept the step as it is too small already
            printf("Step accepted, would be too small otherwise\n");
        }
    }
    else if (*err>def_err && *err<2.0*def_err) {
        // Decrease step but don't recalc
        ujdz = *dzz/err_fact;
        if (ujdz>dzmin) {
            *dzz = ujdz;
            printf("Step decreased, no recalc. \n");
        }
        else {
            // must accept the step as it is too small already
            printf("Step accepted, would be too small otherwise\n");
        }
    }
    else if (*err>0.5*def_err && *err<=def_err) {
        printf("Everything good...\n");
        // Everything good
    }
    else {
        // Increase step
        ujdz = *dzz*err_fact;
        if (ujdz>100*(y2[1]+y1[1])/2.0) // Step won't be bigger than the beam size x 2
            {printf("Step would be too big, keeping it.\n");}
        else {
            *dzz = ujdz;
            printf("Step increased\n");
        }
    }
    //exit(0);
}

