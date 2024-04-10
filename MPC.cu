#define _USE_MATH_DEFINES
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <getopt.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <mpi.h>

#define ZW0 377.0 // free space impedance
#define NC 1.0  // refractive index (set to 1 - air)
#define FCTR 0.89 // factor to account for no electric field oscillations 
#define IH 27.21 // ionization E for H
#define EF 5.14220642e11 // elec. field constant V/m
#define ME 9.1094e-31 // el mass
#define EL0 1.602e-19 // el charge
#define NAVO 6.02214e+23 // Avogadro num
#define T0 2.4188843265e-17  // atomic unit of time  
#define C0 299792458.0 // speed of light
#define HBAR 1.054560652926899e-34 // Planck
#define PROC 16 // num processors to use

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


void ionizpot(const char* gas, double *I1, double *I2);
void ADK(double *ionizationrates1, double *ionizationrates2, int *dims, double *C_nl, double *f_lm, double *nq, double *mq, double *Ei, double *E);
void Natoms(const char *gas, double *Natm, double *sigm_coll);
void calcexpR(cufftDoubleComplex *expR, int *signmem, double z, double Cav, double k0, double dzz, double *wr, int *dims);
void fftshift(cufftDoubleComplex *in, int *dims, cufftDoubleComplex *buffer, int axis);
void ioniz(cufftDoubleComplex *A, int *dims, double w0, double deltat, char* gas, double rho_c, double *w, double rho_nt, double n0, double *tau_c, double *freeelectrons);
void calc1oes (double *BP, int *dims, double *r, double *vald1oes, double *y);
void ssfmprop(cufftDoubleComplex *A, int *dims, int sst_on, double dzz, double *betas, double alpha, int *signmem, double z, double Cav, double k0, double *wr, fftw_plan pR1fft, fftw_plan pR1ifft, fftw_plan pR2fft, fftw_plan pR2ifft, fftw_plan pTfft, fftw_plan pTifft, double *w, double gamma2, double w0, int plasm, double deltat, char *gas, double rho_c, double rho_nt, double n0, double *puls_e, double *r, double *BP1, double *BP2, double *y1, double *y2, double *bps1, double *bps2, double Ab_M, int change, int mpirank);
void linearstep(cufftDoubleComplex *A, cuDoubleComplex *exp_D0, cuDoubleComplex *expR, fftw_plan pTfft, fftw_plan pR2fft, fftw_plan pR1fft, int *dims);
void createPlans(int readin, int *dims, const char* cpath, fftw_plan *pR1fft, fftw_plan *pR1ifft, fftw_plan *pR2fft, fftw_plan *pR2ifft, fftw_plan *pTfft, fftw_plan *pTifft);
void setStep(int *dims, double *dzz, cufftDoubleComplex *C1, cufftDoubleComplex *C2, double def_err, double dzmin, double *y1, double *y2, int *bad, double *err);
void sendmpi(cufftDoubleComplex *A, MPI_Datatype datatype, int *dims, int rankto);
void recmpi(cufftDoubleComplex *A, MPI_Datatype datatype, int *dims, int rankfrom);
void backup(FILE* fp);
void restore(const char* path);
void copyfile(const char* path, int dorestore);
FILE* opener(const char* cpath, const char* fname, const char* fopts, int dorestore);
__global__ void memsetCUDA(double* array, size_t size);
__global__ void spatintegralCUDA(const cufftDoubleComplex *C, double *result, int *dims);
__device__ __forceinline__  double cexpCUDAr(cuDoubleComplex z);
__device__ __forceinline__  double cexpCUDAi(cuDoubleComplex z);
__global__ void calcexpD0CUDA(double dzz, double alpha, double *betas, cuDoubleComplex *exp_D0, int dims);
__global__ void propagT(cuDoubleComplex *A, const cuDoubleComplex *exponent, int *dims);
__global__ void propagR(cuDoubleComplex *A, const cuDoubleComplex *exponent, int *dims, int dir);
__global__ void calcdACUDA(cuDoubleComplex *d_dA, const cuDoubleComplex *d_buffer, const double *d_wbuff, int *dims);
__global__ void calcsstCUDA(cufftDoubleComplex *dA, const cufftDoubleComplex *A, double gamma2, double w0, cufftDoubleComplex *buffer, int dim2, int totaldim);
 

int main(int argc, char *argv[]) {

    int mpirank;
    if (!USEGPU) {
        /* Setting up MPI */
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (provided != MPI_THREAD_FUNNELED) {
            fprintf(stderr, "Warning MPI did not provide MPI_THREAD_FUNNELED\n");
            exit(0);
        }
        int mpisize;
        MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);
        printf("Starting up! hostname: %s, mpirank: %d\n", processor_name, mpirank);
    }
    else {
        /* Using CUDA */
        mpirank = 0;
    }
    
    /* For NEH cluster */
    const char *dir = "/reg/neh/home/kmecseki/broad3/MCP/3D/data";
    const char *cpath2 = "/reg/neh/home/kmecseki/broad3/MCP/3D/data";
    
    char *cpath = (char *) malloc(strlen(dir)+10);
    unsigned int nrun = 0;
    unsigned int resume = 0;
    int cp;

    while ((cp = getopt (argc, argv, "rn:")) != -1)
    switch(cp) {
        case 'r':
            resume = 1;
            printf("Resuming previous run!\n");
            break;
        case 'n':
            nrun = atoi(optarg);
	        break;
        default:
            abort();
    }

    sprintf(cpath, "%s%04d", dir, nrun);

    int dims[3];
    FILE *fparp;
    const int lgt = 80;
        
    /* Open parameter file */
    fparp = opener(cpath, "param.bin", "r", 0);
    fread(dims, sizeof(int), 3, fparp);

    /* Output files */
    FILE *fmarp, *fmaip, *fbp1p, *fbp2p, *ftemp, *fspep, *fothp, *foutp, *fresdp, *frespp;
    char fresdn[lgt], frespn[lgt];
    
    cufftDoubleComplex *C1, *C2;

    if (mpirank == 0) { 
        printf("Using directory %s\n", cpath);
        printf("Matrix dimensions are: %d x %d x %d. \n", dims[0], dims[1], dims[2]);

        C1 = (cufftDoubleComplex *) malloc(dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex));
        C2 = (cufftDoubleComplex *) malloc(dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex));
    
        foutp = opener(cpath, "out.bin",   "wb", 0);
    
        snprintf(fresdn, lgt, "%s/%s", cpath, "resdata.bin");
        snprintf(frespn, lgt, "%s/%s", cpath, "resparam.bin");
    
        if (resume == 1) {
            fbp1p = opener(cpath, "bp1.bin",   "ab",1);
            fbp2p = opener(cpath, "bp2.bin",   "ab",1);
            ftemp = opener(cpath, "temp.bin",  "ab",1);
            fspep = opener(cpath, "spec.bin",  "ab",1);
            fothp = opener(cpath, "other.bin", "a" ,1);
        }
        else {
            fmarp = opener(cpath2, "datar.bin", "r", 0);
            fmaip = opener(cpath2, "datai.bin", "r", 0);
            fbp1p = opener(cpath, "bp1.bin",   "wb",0);
            fbp2p = opener(cpath, "bp2.bin",   "wb",0);
            ftemp = opener(cpath, "temp.bin",  "wb",0);
            fspep = opener(cpath, "spec.bin",  "wb",0);
            fothp = opener(cpath, "other.bin", "w", 0);
        }
    }
    
    char gas[9];
    int sst_on, plasm;

    double Cav, k0;
    fftw_plan pR1fft, pR1ifft, pR2fft, pR2ifft, pTfft, pTifft;

    double gamma2, w0, deltat, rho_c, rho_nt, n0, puls_e, Ab_M;
    double bps1 = 0.0, bps2 = 0.0;
    double y1[4], y2[4];
    
    if ((mpirank == 0)  && (!USEGPU)) {
        printf("DEBUG: Allocating memory on all nodes...\n");
    }

    cufftDoubleComplex *A;
    double *betas, *wr, *w, *r, *BP1, *BP2, *temp, *spec;
    A = (cufftDoubleComplex *)malloc(dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex));
    betas = (double *)malloc(dims[2] * sizeof(double));
    wr = (double *)malloc(dims[1] * sizeof(double));
    w = (double *)malloc(dims[2] * sizeof(double));
    r = (double *)malloc(dims[1] * sizeof(double));
    BP1 = (double *)malloc(dims[0] * sizeof(double));
    BP2 = (double *)malloc(dims[1] * sizeof(double));
    
    if (mpirank == 0) {

        temp = (double *)malloc(dims[2] * sizeof(double));
        spec = (double *)malloc(dims[2] * sizeof(double));

        if (resume == 1) {
            printf("Reading resume data matrix!\n");
            fresdp = fopen(fresdn,"r");
            fread(A, sizeof(cufftDoubleComplex), dims[0]*dims[1]*dims[2], fresdp);
            fclose(fresdp);
            printf("Read resume data matrix successfully!\n");
        }
        else {
            double *Ar, *Ai;
            Ar = (double *)malloc(dims[0] * dims[1] * dims[2] * sizeof(double));
            Ai = (double *)malloc(dims[0] * dims[1] * dims[2] * sizeof(double));
            temp = (double *)malloc(dims[2] * sizeof(double));
            spec = (double *)malloc(dims[2] * sizeof(double));
    
            printf("Now reading in data matrix...\n");
            fread(Ar, sizeof(double), dims[0] * dims[1] * dims[2], fmarp);
            fread(Ai, sizeof(double), dims[0] * dims[1] * dims[2], fmaip);
            fclose(fmaip);
            fclose(fmarp);
        
            #pragma omp parallel for
            for (int i=0; i<dims[0]*dims[1]*dims[2]; ++i) {
                A[i].x = Ar[i];
                A[i].y = Ai[i];
            }
            free(Ar);
            free(Ai);
            printf("\bDone!\n");
        }  
    }
    if (mpirank == 0) {
        printf("Now reading in parameters...\n");
    }

    /* reading input parameters */
    double dist, dzz, alpha, def_err, dzmin, dr, Esig, Ksn;
    fread(&dist, sizeof(double), 1, fparp);
    fread(&dzz, sizeof(double), 1, fparp);
    fread(betas, sizeof(double), dims[2], fparp);
    fread(&alpha, sizeof(double), 1, fparp);
    fread(&Cav, sizeof(double), 1, fparp);
    fread(&k0, sizeof(double), 1, fparp);
    fread(wr, sizeof(double), dims[1], fparp);
    fread(&sst_on, sizeof(int), 1, fparp);
    fread(w, sizeof(double), dims[2], fparp);
    fread(&gamma2, sizeof(double), 1, fparp);
    fread(&w0, sizeof(double), 1, fparp);
    fread(&plasm, sizeof(int), 1, fparp);
    fread(&deltat, sizeof(double), 1, fparp);
    fread(&rho_c, sizeof(double), 1, fparp);
    fread(&rho_nt, sizeof(double), 1, fparp);
    fread(&n0, sizeof(double), 1, fparp);
    fread(&puls_e, sizeof(double), 1, fparp);
    fread(&Ab_M, sizeof(double), 1, fparp);
    fread(r, sizeof(double), dims[1], fparp);
    fread(&dr, sizeof(double), 1, fparp);
    fread(&def_err, sizeof(double), 1, fparp);
    fread(&dzmin, sizeof(double), 1, fparp);
    fread(&Esig, sizeof(double),1,fparp);
    fread(gas, sizeof(char), 9, fparp);
    fclose(fparp);
    
    if (mpirank == 0) {
        printf("\bDone!\n");
        printf("Calibrating energy...\n");
        Ksn = sqrt(puls_e/Esig);

        #pragma omp parallel for
        for (int i=0; i<dims[0]*dims[1]*dims[2]; i++) {
            A[i].x = Ksn * A[i].x;
        }
        printf("\bDone!\n");
    }
    
    if (mpirank == 0) {
        printf("Creating FFT plans...\n");
    }
    
    #if USEGPU == 0
    /* use all processors to speed this up a bit */
    fftw_init_threads();
    fftw_plan_with_nthreads(PROC);
    #endif

    createPlans(READWIS, dims, cpath2, &pR1fft, &pR1ifft, &pR2fft, &pR2ifft, &pTfft, &pTifft);
    
    if (mpirank == 0) {
        printf("\bDone!\n");
    }
    
    double err = 0; // track error
    double z = 0; // propagation distance
    int bad = 0;
    int signmem = 1; // changes as the pulse travels back and forth
    int nstep = 0;

    if ((mpirank == 0) && (!USEGPU)) {
        if (resume == 1) {
            printf("Reading resume parameters!\n");
            frespp = fopen(frespn,"r");
            fread(&dzz, sizeof(double), 1, frespp);
            fread(&z, sizeof(double), 1, frespp);
            fread(&signmem, sizeof(int), 1, frespp);
            fread(&puls_e, sizeof(double), 1, frespp);
            fread(&nstep, sizeof(int), 1, frespp);
            fclose(frespp);
            printf("Resume parameters read!\n");
        }
    }

    if (!USEGPU) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpirank == 0) {
            printf("\bAll nodes are ready, for loop is starting!\n");
        }
    }
    else {
        printf("\bStarting main loop using CUDA!\n");
    }
    
    // DEBUG:
    // clock_t start, end;
    // double cpu_time_used;

    /* MAIN LOOP STARTS HERE */
    while (z < dist) {
               
        if ((mpirank == 0) && (!USEGPU)) {
            //start = clock();
            // printf("DEBUG: Sending values from rank 0!\n");
            sendmpi(A, MPI_C_DOUBLE_COMPLEX, dims, 1);           
            // printf("DEBUG: One down!\n");
            sendmpi(A, MPI_C_DOUBLE_COMPLEX, dims, 3);
            // printf("DEBUG: A sent to 1 and 3!\n");
            
            for (int i = 1; i < 5; ++i) {
                MPI_Send(&dzz, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(&z, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(&signmem, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&puls_e, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
            
            MPI_Recv(&signmem, 1, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recmpi(C1, MPI_C_DOUBLE_COMPLEX, dims, 1);
            recmpi(C2, MPI_C_DOUBLE_COMPLEX, dims, 3);
            MPI_Recv(BP1, dims[0], MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(BP2, dims[1], MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&y1, 4, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&y2, 4, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (mpirank > 0 && mpirank < 3) {
            /* rank 1 and 2 */
            if (mpirank == 1) {
                recmpi(A, MPI_C_DOUBLE_COMPLEX, dims, 0);      
            }

            MPI_Recv(&dzz, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&z, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&signmem, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&puls_e, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            /* Double step for C1 */
            ssfmprop(A, dims, sst_on, 2*dzz, betas, alpha, &signmem, z, Cav, k0, wr, pR1fft, pR1ifft, pR2fft, pR2ifft, pTfft, pTifft, w, gamma2, w0, plasm, deltat, gas, rho_c, rho_nt, n0, &puls_e, r, BP1, BP2, y1, y2, &bps1, &bps2, Ab_M, 0, mpirank);
            
            if (mpirank == 1) {
                sendmpi(A, MPI_C_DOUBLE_COMPLEX, dims, 0);
            }
        }
       
        if (mpirank>2) {
            /* for rank 3 and 4 */
            if (mpirank == 3) {
                recmpi(A, MPI_C_DOUBLE_COMPLEX, dims, 0);
            }

            MPI_Recv(&dzz, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&z, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&signmem, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&puls_e, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            /* two steps for C2 */
            ssfmprop(A, dims, sst_on, dzz, betas, alpha, &signmem, z, Cav, k0, wr, pR1fft, pR1ifft, pR2fft, pR2ifft, pTfft, pTifft, w, gamma2, w0, plasm, deltat, gas, rho_c, rho_nt, n0, &puls_e, r, BP1, BP2, y1, y2, &bps1, &bps2, Ab_M, 0, mpirank);
            ssfmprop(A, dims, sst_on, dzz, betas, alpha, &signmem, z, Cav, k0, wr, pR1fft, pR1ifft, pR2fft, pR2ifft, pTfft, pTifft, w, gamma2, w0, plasm, deltat, gas, rho_c, rho_nt, n0, &puls_e, r, BP1, BP2, y1, y2, &bps1, &bps2, Ab_M, 1, mpirank);
            if (mpirank == 3) {
                MPI_Send(&signmem, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                sendmpi(A, MPI_C_DOUBLE_COMPLEX, dims, 0);
                MPI_Send(BP1, dims[0], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(BP2, dims[1], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&y1, 4, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&y2, 4, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }

        if (!USEGPU) {
            MPI_Barrier(MPI_COMM_WORLD);
        }
        else {                
            /* advance doouble step for C1 */
            ssfmprop(C1, dims, sst_on, 2*dzz, betas, alpha, &signmem, z, Cav, k0, wr, pR1fft, pR1ifft, pR2fft, pR2ifft, pTfft, pTifft, w, gamma2, w0, plasm, deltat, gas, rho_c, rho_nt, n0, &puls_e, r, BP1, BP2, y1, y2, &bps1, &bps2, Ab_M, 0, mpirank);
            
            /* advance single step twice for C2 */
            ssfmprop(C2, dims, sst_on, dzz, betas, alpha, &signmem, z, Cav, k0, wr, pR1fft, pR1ifft, pR2fft, pR2ifft, pTfft, pTifft, w, gamma2, w0, plasm, deltat, gas, rho_c, rho_nt, n0, &puls_e, r, BP1, BP2, y1, y2, &bps1, &bps2, Ab_M, 0, mpirank);
            ssfmprop(C2, dims, sst_on, dzz, betas, alpha, &signmem, z, Cav, k0, wr, pR1fft, pR1ifft, pR2fft, pR2ifft, pTfft, pTifft, w, gamma2, w0, plasm, deltat, gas, rho_c, rho_nt, n0, &puls_e, r, BP1, BP2, y1, y2, &bps1, &bps2, Ab_M, 0, mpirank);
        }

        if (mpirank == 0) {
            /* correction for step size */
            setStep(dims, &dzz, C1, C2, def_err, dzmin, y1, y2, &bad, &err);
            
            //DEBUG:
            // if (dr>(y1[1]+y2[1])/2.0) {
            //     printf("ERROR Rdim too small, nonlinear effects pull the beam too small for this resolution.\n Specs: y1 = %f, y2 = %f, dr = %f\n ",y1[1], y2[1], dr);
            //     exit(0);
            // }
            if (!USEGPU) {
                /* broadcast if the step needs voided */
                for (int i = 1; i < 5; ++i) {
                    MPI_Send(&bad, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
            }
        }

        if (mpirank > 0) {
            MPI_Recv(&bad, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (bad == 1) {
            /* skip saving this step if the error is too much */
            continue;
        }

        else {
            if (mpirank == 0) {
                #pragma omp parallel for 
                for (int i=0; i<dims[2]*dims[1]*dims[0]; ++i) {
                    A[i].x = (4.0 / 3.0) * C2[i].x - (1.0 / 3.0) * C1[i].x;
                    A[i].y =  (4.0 / 3.0) * C2[i].y  - (1.0 / 3.0) * C1[i].y;
                }
            
                z = z + 2.0 * dzz;
                memset(temp, 0, dims[2] * sizeof(double));


                /* get temp profile and spec */
                    
                #if USEGPU == 1

                int *d_dims;
                cudaMalloc((void**)&d_dims, 3 * sizeof(int));
                cudaMemcpy(d_dims, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);

                cufftDoubleComplex *d_C2;
                cudaMalloc((void**)&d_C2, sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2]);
                cudaMemcpy(d_C2, A, dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
                
                double *d_temp;
                cudaMalloc((void**)&d_temp, sizeof(cufftDoubleComplex) * dims[2]);

                int tpb = 256;
                int bpg = (dims[2] + tpb - 1) / tpb;

                /* set to 0 */
                memsetCUDA<<<bpg, tpb>>>(d_temp, dims[2]);
                cudaError_t cerr = cudaGetLastError();
                if (cerr != cudaSuccess) {
                    fprintf(stderr, "CUDA error when setting temp array to 0: %s\n", cudaGetErrorString(cerr));
                }
                cudaDeviceSynchronize();

                spatintegralCUDA<<<bpg, tpb>>>(d_C2, d_temp, d_dims);
                cerr = cudaGetLastError();
                if (cerr != cudaSuccess) {
                    fprintf(stderr, "CUDA error when calculating temporal profile: %s\n", cudaGetErrorString(cerr));
                }
                cudaDeviceSynchronize();

                /* FFT */
                cufftExecZ2Z(pTfft, d_C2, d_C2, CUFFT_FORWARD);

                double *d_spec;
                cudaMalloc((void**)&d_spec, sizeof(cufftDoubleComplex) * dims[2]);

                /* set to 0 */
                memsetCUDA<<<bpg, tpb>>>(d_spec, dims[2]);
                cerr = cudaGetLastError();
                if (cerr != cudaSuccess) {
                    fprintf(stderr, "CUDA error when setting spec array to 0: %s\n", cudaGetErrorString(cerr));
                }
                cudaDeviceSynchronize();

                spatintegralCUDA<<<bpg, tpb>>>(d_C2, d_spec, d_dims);
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    fprintf(stderr, "CUDA error when calculating spectrum: %s\n", cudaGetErrorString(cerr));
                }
                cudaDeviceSynchronize();

                /* Get results */
                cudaMemcpy(temp, d_temp, dims[2] * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
                cudaMemcpy(spec, d_spec, dims[2] * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

                cudaFree(d_dims);
                cudaFree(d_C2);
                cudaFree(d_spec);
                cudaFree(d_temp);

                #else
                #pragma omp parallel sections
                {
                    #pragma omp section
                    {
                        #pragma omp parallel for
                        for (int k=0; k<dims[2]; k++) {
                            //#pragma omp parallel for private(i,k) shared(temp) reduction(+: temp)
                            for (int i = 0; i < dims[1] * dims[0]; i++) {
                                temp[k] += A[k * dims[1] * dims[0] + i];
                            }
                            temp[k] = cabs(temp[k]);
                        }
                    }

                    #pragma omp section
                    {
                        memcpy(C2, A, dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex));
                        memset(spec, 0, dims[2] * sizeof(double));

                        fftw_execute_dft(pTfft, C2, C2);

                        #pragma omp parallel for
                        for (int k=0; k<dims[2]; k++) {
                            //#pragma omp parallel for private(i,k) shared(spec) reduction(+: spec)
                            for (int i=0; i<dims[1]*dims[0]; i++) {
                                spec[k] += C2[k*dims[1]*dims[0]+i];                                
                            }
                            spec[k] = cabs(spec[k])
                        }
                    }
                }
                #endif
                
            
                #pragma omp parallel sections 
                {
                    #pragma omp section 
                    {
                        fwrite(BP1, sizeof(double), dims[0], fbp1p);
                        fflush(fbp1p);
                    }
                    #pragma omp section 
                    {
                        fwrite(BP2, sizeof(double), dims[1], fbp2p);
                        fflush(fbp2p);
                    }
                    #pragma omp section 
                    {
                        fwrite(temp, sizeof(double), dims[2], ftemp); 
                        fflush(ftemp);
                    }
                    #pragma omp section 
                    {
                        fwrite(spec, sizeof(double), dims[2], fspep);
                        fflush(fspep);
                    }
                    #pragma omp section 
                    {
                        fprintf(fothp, "%f\t%f\n", err, dzz);
                        fflush(fothp);
                    }
                }

                if (z > dist) {
                    /* We are at the end, do not overshoot */
                    z = dist;
                }

                printf("We are here: %5.2f\n", z * 100.0 / dist );
                        
                if (mpirank == 0) {
                    if ((nstep%10) == 0) {
                        printf("We are at step %d, writing checkpoint data into files.\n", nstep );
                    
                        int stepn;
                        stepn = nstep + 1;
                        fresdp = fopen(fresdn, "wb");
                        fwrite(A, sizeof(cufftDoubleComplex), dims[0] * dims[1] * dims[2], fresdp);
                        fclose(fresdp);
                        frespp = fopen(frespn, "wb");
                        fwrite(&dzz, sizeof(double), 1, frespp);
                        fwrite(&z, sizeof(double), 1, frespp);
                        fwrite(&signmem, sizeof(int), 1, frespp);
                        fwrite(&puls_e, sizeof(double), 1, frespp);
                        fwrite(&stepn, sizeof(int), 1, frespp);
                                      
                        fclose(frespp);
                    
                        backup(fbp1p);
                        backup(fbp2p);
                        backup(ftemp);
                        backup(fspep);
                        backup(fothp);
                    }
                    // DEBUG:
		            //end = clock();
		            //cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		            //printf("One step took %f\n", cpu_time_used);
                }
                
                nstep++;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (mpirank == 0) { 
        fwrite(A, sizeof(double), 2 * dims[0] * dims[1] * dims[2], foutp);
        fclose(fbp1p);
        fclose(fbp2p);
        fclose(ftemp);
        fclose(fspep);
        fclose(fothp);
        fclose(foutp);
    }
     
    fftw_destroy_plan(pR1fft);
    fftw_destroy_plan(pR1ifft);
    fftw_destroy_plan(pR2fft);
    fftw_destroy_plan(pR2ifft);
    fftw_destroy_plan(pTfft);
    fftw_destroy_plan(pTifft);
    #if USEGPU == 0
    fftw_cleanup(); 
    fftw_cleanup_threads();
    #endif

    free(A);
    free(betas);
    free(wr);
    free(w);
    free(r);
    free(BP1);
    free(BP2);
    free(cpath);
      
    if (mpirank == 0) { 
        free(C1);
        free(C2);
        free(temp);
        free(spec);
    }

    MPI_Finalize();
    return 0;
}    


void backup(FILE* fp) {
    int fd;
    char* buffer = (char *)malloc(PATH_MAX);
    char* path = (char *)malloc(PATH_MAX);
    
    fd = fileno(fp);
    snprintf(buffer, PATH_MAX, "/proc/self/fd/%d", fd);
    memset(path, 0, PATH_MAX);
    readlink(buffer, path, PATH_MAX-1);
    
    free(buffer);
    
    copyfile(path, 0);
    
    free(path);
}


void restore(const char* path) {
    copyfile(path, 1);
}


void copyfile(const char* path, int dorestore) {

    char* buffer = (char *)malloc(2 * strlen(path) + 9);
    if (!buffer) {
        fprintf(stderr,"Memory allocation failed in copyfile function");
        free(buffer);
        exit(1);
    }
    if (dorestore) {
        sprintf(buffer, "cp %s.bak %s", path, path);
    } else {
        sprintf(buffer, "cp %s %s.bak", path, path);
    }
    system(buffer);
    if (system(buffer) == -1) {
        fprintf(stderr,"Error executing system command");
        free(buffer);
        exit(1);
    }
    free(buffer);
}


FILE* opener(const char* cpath, const char* fname, const char* fopts, int dorestore) {
    char* path;
    FILE* fp;
    
    // create pathname and open file
    path = (char *)malloc(strlen(cpath) + strlen(fname) + 2);
    if (!path) {
        fprintf(stderr, "Memory allocation failed in opener function\n");
        exit(1);
    }
    sprintf(path, "%s/%s", cpath, fname);
    if (dorestore) restore(path);
    fp = fopen(path, fopts);
    if (!fp) {
        fprintf(stderr, "Failed to open file: %s\n", path);
        exit(1);
    }
    free(path);
    return fp;
}


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


void linearstep(cufftDoubleComplex *A, cuDoubleComplex *exp_D0, cuDoubleComplex *expR, fftw_plan pTfft, fftw_plan pR2fft, fftw_plan pR1fft, int *dims) {
    #if USEGPU == 1

    cufftDoubleComplex *d_A;
    cudaMalloc((void**)&d_A, sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2]);
    cudaMemcpy(d_A, A, dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    /* FFT */
    cufftExecZ2Z(pTfft, d_A, d_A, CUFFT_FORWARD);

    int *d_dims;
    cudaMalloc((void **)&d_dims, 3 * sizeof(int));
    cudaMemcpy(d_dims, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);

    cuDoubleComplex *d_exp_D0;
    cudaMalloc((void **)&d_exp_D0, dims[2] * sizeof(cuDoubleComplex));
    cudaMemcpy(d_exp_D0, exp_D0, dims[2] * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    /* one block per temporal axis */
    propagTCUDA<<<dims[2], dims[0] * dims[1]>>>(d_A, d_exp_D0, d_dims);
    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "CUDA error when calculating spectrum: %s\n", cudaGetErrorString(cerr));
    }
    cudaDeviceSynchronize();

    /* FFT */
    cufftExecZ2Z(pTfft, d_A, d_A, CUFFT_INVERSE);
    cufftExecZ2Z(pR2fft, d_A, d_A, CUFFT_FORWARD);
        
    cufftDoubleComplex *d_expR;
    cudaMalloc((void**)&d_expR, sizeof(cufftDoubleComplex) * dims[0]);
    cudaMemcpy(d_expR, expR, dims[0] * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    dim3 tpb2(dims[0], dims[1]);
    int bpg = dims[2]; // One block per k
    propagRCUDA<<<bpg, tpb2>>>(d_A, d_expR, dims, 1);
    cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "CUDA error when calculating spectrum: %s\n", cudaGetErrorString(cerr));
    }
    cudaDeviceSynchronize();

    /* FFT */
    cufftExecZ2Z(pR2fft, d_A, d_A, CUFFT_INVERSE);
    cufftExecZ2Z(pR1fft, d_A, d_A, CUFFT_FORWARD);

    propagRCUDA<<<bpg, tpb2>>>(d_A, d_expR, dims, 2);
    cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "CUDA error when calculating spectrum: %s\n", cudaGetErrorString(cerr));
    }
    cudaDeviceSynchronize();

    cufftExecZ2Z(pR1fft, d_A, d_A, CUFFT_INVERSE);

    cudaMemcpy(A, d_A, dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

    cudaFree(d_expR);
    cudaFree(d_dims);
    cudaFree(d_A);

    #else
        
    fftw_execute_dft(pTfft, A, A);

    #pragma omp parallel for collapse(2)
    for (int k=0; k<dims[2]; k++) {
        for (int i=0; i<dims[0]*dims[1]; i++) {
            A[k * dims[0] * dims[1] + i] = A[k * dims[0] * dims[1] + i] * exp_D0[k];
        }
    }

    fftw_execute_dft(pTifft, A, A);
    fftw_execute_dft(pR2fft, A, A);

    #pragma omp simd collapse(3) 
    for (int k=0; k<dims[2]; k++) {
        for (int j=0; j<dims[1]; j++) {
            for (int i=0; i<dims[0]; i++) {
                A[k * dims[0] * dims[1] + j * dims[1] + i] = A[k * dims[0] * dims[1] + j * dims[1] + i] * conj(expR[j]);
            }
        }
    }  

    fftw_execute_dft(pR2ifft, A, A);
    fftw_execute_dft(pR1fft, A, A);

    #pragma omp simd collapse(3) 
    for (int k=0; k<dims[2]; k++) {
        for (int j=0; j<dims[1]; j++) {
            for (int i=0; i<dims[0]; i++) {
                A[k * dims[0] * dims[1] + j * dims[1] + i] = A[k * dims[0] * dims[1] + j * dims[1] + i] * conj(expR[i]);
            }
        }
    }

    fftw_execute_dft(pR1ifft, A, A);
    #endif
    
    #pragma omp parallel for  
    for (int k=0; k<dims[2]*dims[1]*dims[0]; k++) {
        A[k].x = A[k].x / ((double) (dims[2] * dims[1] * dims[0]));
        A[k].y = A[k].y / ((double) (dims[2] * dims[1] * dims[0]));
    }
}


void ssfmprop(cufftDoubleComplex *A, int *dims, int sst_on, double dzz, double *betas, double alpha, int *signmem, double z, double Cav, double k0, double *wr, fftw_plan pR1fft, fftw_plan pR1ifft, fftw_plan pR2fft, fftw_plan pR2ifft, fftw_plan pTfft, fftw_plan pTifft, double *w, double gamma2, double w0, int plasm, double deltat, char* gas, double rho_c, double rho_nt, double n0, double *puls_e, double *r, double *BP1, double *BP2, double *y1, double *y2, double *bps1, double *bps2, double Ab_M, int change, int mpirank) {   
   
    int signum;
    double *freeelectrons_sp;
    cufftDoubleComplex *exp_D0, *buffersmall, *expR, *buffer, *dA, *A_nl;
    double *tau_c;
    A_nl = (cufftDoubleComplex *)malloc(dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex));
    buffer = (cufftDoubleComplex *)malloc(dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex));
    
    if (mpirank == 1 || mpirank == 3 || USEGPU){
        
        exp_D0 = (cufftDoubleComplex*)malloc(dims[2] * sizeof(cufftDoubleComplex));
        expR = (cufftDoubleComplex*)malloc(dims[0] * sizeof(cufftDoubleComplex));
        buffersmall = (cufftDoubleComplex *)malloc(dims[2] * sizeof(cufftDoubleComplex));
        dA = (cufftDoubleComplex *)malloc(dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex));
        
        /* Linear step - half step */
        #if USEGPU == 1

        double *d_betas;
        cuDoubleComplex *d_exp_D0;
        cudaMalloc((void **)&d_betas, dims[2] * sizeof(double));
        cudaMalloc((void **)&d_exp_D0, dims[2] * sizeof(cuDoubleComplex));
        cudaMemcpy(d_betas, betas, dims[2] * sizeof(double), cudaMemcpyHostToDevice);

        int tpb = 256;
        int bpg = (dims[2] + tpb -1) / tpb;
        calcexpD0CUDA<<<bpg, tpb>>>(dzz, alpha, d_betas, d_exp_D0, dims[2]);
        cudaMemcpy(exp_D0, d_exp_D0, dims[2] * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

        cudaError_t cerr = cudaGetLastError();
        if (cerr != cudaSuccess) {
            fprintf(stderr, "CUDA error when calculating spectrum: %s\n", cudaGetErrorString(cerr));
        }
        cudaDeviceSynchronize();
        cudaFree(d_betas);
        cudaFree(d_exp_D0);

        #else
        #pragma omp parallel for
        for (int i=0; i<dims[2]; ++i) {
            exp_D0[i] = cexp(dzz / 2.0 * (I * betas[i] - alpha / 2.0));
        }
        #endif

        fftshift(exp_D0, dims, buffersmall, 0);   

        calcexpR(expR, signmem, z, Cav, k0, dzz / 2, wr, dims); // parallel replacement for above

        linearstep(A, exp_D0, expR, pTfft, pR2fft, pR1fft, dims);

        /* Nonlinear step - full step */
        
        if (sst_on == 1) {
            #pragma omp parallel for
            for (int i=0; i<dims[0]*dims[1]*dims[2]; i++) {
               buffer[i] = A[i];
            }
     
            double *wbuff;
            wbuff = (double *)malloc(dims[2] * sizeof(double));
            memcpy (wbuff, w, dims[2] * sizeof(double) );
            #pragma omp parallel for
            for (int k=0; k<dims[2]/2; k++) {
                wbuff[k] = w[dims[2]/2+k];
                wbuff[dims[2]/2+k] = w[k];
            }

            #if USEGPU == 1
            cufftDoubleComplex *d_buffer;
            cudaMalloc((void **)&d_buffer, dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex));
            cudaMemcpy(d_buffer, buffer, dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

            cufftExecZ2Z(pR2fft, d_buffer, d_buffer, CUFFT_FORWARD);

            cufftDoubleComplex *d_dA;
            cudaMalloc((void **)&d_dA, dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex));
            double *d_wbuff;
            cudaMalloc((void **)&d_wbuff, dims[2] * sizeof(double));
            cudaMemcpy(d_wbuff, wbuff, dims[2] * sizeof(double), cudaMemcpyHostToDevice);
            //cudaMemcpy(buffer, d_buffer, dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
            int *d_dims;
            cudaMalloc((void **)&d_dims, 3 * sizeof(int));
            cudaMemcpy(d_dims, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);
            
            calcdACUDA<<<dims[2], dims[0] * dims[1]>>>(d_dA, d_buffer, d_wbuff, dims);
            
            cufftExecZ2Z(pTfft, d_dA, d_dA, CUFFT_INVERSE);

            cudaFree(d_wbuff);
            cudaFree(d_dims);
            
            cufftDoubleComplex *d_A;
            cudaMalloc((void**)&d_A, sizeof(cufftDoubleComplex) * dims[0] * dims[1] * dims[2]);
            cudaMemcpy(d_A, A, dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
            int tpb = 256;
            int bpg = (dims[0] * dims[1] * dims[2] + 256 - 1) / 256;
            calcsstCUDA<<<bpg, tpb>>>(d_dA, d_A, gamma2, w0, d_buffer, dims[2], dims[0] * dims[1] * dims[2]);

            cudaMemcpy(A, d_A, dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
            cudaFree(d_A);
            cudaFree(d_dA);
            cudaFree(d_buffer);

            #else
            fftw_execute_dft(pTfft, buffer, buffer);
            #pragma omp parallel for collapse(2)
            for (int k=0; k<dims[2]; k++) {
                for (int i=0; i<dims[0]*dims[1]; i++) {
                    dA[k * dims[0] * dims[1] + i] = -1.0 * I * buffer[k * dims[0] * dims[1] + i] * wbuff[k];
                }
            }
            fftw_execute_dft(pTifft, dA, dA);

            #pragma omp parallel for
            for (int k=0; k<dims[2]*dims[1]*dims[0]; k++) {
                dA[k] = dA[k] / ((double) (dims[2]));
            }

            #pragma omp parallel for  
            for (int i=0; i<dims[0]*dims[1]*dims[2]; i++) {
                buffer[i] = gamma2/w0*(2.0*dA[i]*conj(A[i])+A[i]*conj(dA[i])); // Re-using buffer to save memory
            }

            #endif

            free(wbuff);
        }
        else {
            memset(buffer, 0, dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex));
        }
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    /* Plasma physics */
    if (plasm == 1) {
        #if USEGPU == 0
        if (mpirank == 1 || mpirank == 3) {
            sendmpi(A, MPI_C_DOUBLE_COMPLEX, dims, mpirank + 1);
            sendmpi(buffer, MPI_C_DOUBLE_COMPLEX, dims, mpirank + 1);

            MPI_Send(dims, 3, MPI_INT, mpirank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(&w0, 1, MPI_DOUBLE, mpirank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(&deltat, 1, MPI_DOUBLE, mpirank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(gas, 9, MPI_CHAR, mpirank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(&rho_c, 1, MPI_DOUBLE, mpirank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(w, dims[2], MPI_DOUBLE, mpirank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(&rho_nt, 1, MPI_DOUBLE, mpirank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(&n0, 1, MPI_DOUBLE, mpirank + 1, 0, MPI_COMM_WORLD);
            
            recmpi(A_nl, MPI_C_DOUBLE_COMPLEX, dims, mpirank+1);
        }
        #endif
        if (mpirank == 2 || mpirank == 4 || USEGPU) {
            #if USEGPU == 1
            tau_c = (double *)malloc(dims[0] * dims[1] * dims[2] * sizeof(double));
            freeelectrons_sp = (double *)malloc(dims[0] * dims[1] * dims[2] * sizeof(double));
            #else
            recmpi(A, MPI_C_DOUBLE_COMPLEX, dims, mpirank-1);
            
            recmpi(buffer, MPI_C_DOUBLE_COMPLEX, dims, mpirank-1);
            MPI_Recv(dims, 3, MPI_INT, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&w0, 1, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&deltat, 1, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(gas, 9, MPI_CHAR, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&rho_c, 1, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(w, dims[2], MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&rho_nt, 1, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&n0, 1, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            #endif
            ioniz(A, dims, w0, deltat, gas, rho_c, w, rho_nt, n0, tau_c, freeelectrons_sp);
            
            cuDoubleComplex thei = make_cuDoubleComplex(0.0, 1.0);
            cuDoubleComplex z1, z2;
            #pragma omp parallel for
            for (int i=0; i<dims[0]*dims[1]*dims[2]; i++) {
                z1.x = thei.x * gamma2 * pow(cuCabs(A[i]),2.0);
                z1.y = thei.y * gamma2 * pow(cuCabs(A[i]),2.0);
                z2.x = (1.0 + thei.x * w0 * tau_c[i]) * rho_nt * freeelectrons_sp[i]/2.0;
                z2.y = (1.0 + thei.y * w0 * tau_c[i]) * rho_nt * freeelectrons_sp[i]/2.0;
                A_nl[i].x = z1.x - buffer[i].x - z2.x;
                A_nl[i].y = z1.y - buffer[i].y - z2.y;
            }
            #if USEGPU == 0
            sendmpi(A_nl, MPI_C_DOUBLE_COMPLEX, dims, mpirank-1);
            #endif

            free(tau_c);
            free(freeelectrons_sp);
        }
    }
    else {
        cuDoubleComplex thei = make_cuDoubleComplex(0.0, 1.0);
        #pragma omp parallel for  
        for (int i=0; i<dims[0]*dims[1]*dims[2]; i++) {
            double tempab = pow(cuCabs(A[i]),2.0);
            A_nl[i].x = thei.x * gamma2 * tempab - buffer[i].x;
            A_nl[i].y = thei.y * gamma2 * tempab - buffer[i].y;
        }
    }
    //printf("Waiting for everyone, rank: %d\n", mpirank);
    //MPI_Barrier(MPI_COMM_WORLD);
    //printf("Everybody done\n");
    
    if (mpirank == 1 || mpirank == 3 || USEGPU) {

        #pragma omp parallel for
        for (int i=0; i<dims[0]*dims[1]*dims[2]; i++) {
            double exp_real = exp(dzz * A_nl[i].x);
            double exp_imag = dzz * A_nl[i].y;

            double cos_imag = cos(exp_imag);
            double sin_imag = sin(exp_imag);

            A[i].x = A[i].x * exp_real * cos_imag - A[i].y * exp_real * sin_imag;
            A[i].y = A[i].y * exp_real * sin_imag + A[i].x * exp_real * cos_imag;
        }

        /* Linear step - half step */
        // printf("Linear step...\n");

        linearstep(A, exp_D0, expR, pTfft, pR2fft, pR1fft, dims);
           
        if (change == 1) {
            signum = ((int) (floor(z/Cav) + 1) % 2) * 2-1;
            if (*signmem == -signum) {
                /* account for some absorption */
                *puls_e = *puls_e * Ab_M;
                *signmem *= -1;
            }
        }

        if (change == 1) {
            memset(BP1, 0, dims[0] * sizeof(double));
            memset(BP2, 0, dims[1] * sizeof(double));
            //#pragma omp parallel for collapse(3) shared(BP1,BP2) reduction(+:BP1,BP2)
            for (int i=0; i<dims[2]; i++) {
                for (int j=0; j<dims[1]; j++) {
                    for (int k=0; k<dims[0]; k++) {
                        BP1[k] += pow(cuCabs(A[k + dims[0] * j + i * dims[0] * dims[1]]),2.0);
                        BP2[j] += pow(cuCabs(A[k + dims[0] * j + i * dims[0] * dims[1]]),2.0);
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
    
        free(dA);
        free(exp_D0);
        free(expR);
        free(buffersmall);
    }

    free(A_nl);
    free(buffer);
}


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


void calcexpR(cufftDoubleComplex *expR, int *signmem, double z, double Cav, double k0, double dzz, double *wr, int *dims) {
    
    int signum;
    signum = ((int)(floor(z/Cav) + 1) % 2) * 2 - 1;
    
    if (-*signmem == signum) {
        *signmem = -*signmem;
    }

    #pragma omp parallel for
    for (int i=0; i<dims[0]; i++) {
        double realPart = signum * (-1.0 / (2.0 * k0) * dzz * (-1.0)) * pow(wr[i] - (wr[dims[1] / 2 - 1] + wr[dims[1] / 2]) / 2.0, 2);
        expR[i].x = exp(realPart) * cos(0.0);
        expR[i].y = exp(realPart) * sin(0.0);
        //expR[i] = cexp(signum * (-I/(2.0 * k0) * dzz * (-1.0)) * pow(wr[i] - (wr[dims[1] / 2-1] + wr[dims[1] / 2]) / 2.0, 2));
    }
}


void fftshift(cufftDoubleComplex *in, int *dims, cufftDoubleComplex *buffer, int axis) {

    if (axis == 0) { //Special case for exp_D0
        memcpy ( buffer, in, dims[2] * sizeof(cufftDoubleComplex) );
        #pragma omp parallel for
        for (int i=0; i<dims[2]/2; i++) {
            in[i] = buffer[dims[2]/2+i];
            in[dims[2]/2+i] = buffer[i];
        }
    }            
    else {
        memcpy ( buffer, in, dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex) );
        #pragma omp parallel num_threads(PROC)
        {
            if (axis == 1) {
                #pragma omp parallel for collapse(3)
                for (int k=0; k<dims[2]; k++) {
                    for (int j=0; j<dims[1]; j++) {
                        for (int i=0; i<(dims[0]/2); i++) {
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
            if (axis == 2) {
                #pragma omp parallel for collapse(3)
                for (int k=0; k<dims[2]; k++) {
                    for (int j=0; j<dims[1]/2; j++) {
                        for (int i=0; i<(dims[0]); i++) {
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
            if (axis == 3) {
                #pragma omp parallel for collapse(3)
                for (int k=0; k<dims[2]/2; k++) {
                    for (int j=0; j<dims[1]; j++) {
                        for (int i=0; i<(dims[0]); i++) {
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
    
    /* 1/e^2 beam size calc */
    double bpex = 0;
    int posi1 = 0, posi2 = 0;
    double *normBP, halfBP;
    normBP = (double *)malloc(dims[0] * sizeof(double));

    memcpy(normBP, BP, dims[0] * sizeof(double));
    #pragma omp parallel for
    for (int k=0; k<dims[0]; k++) {
       if (BP[k] > bpex)
           bpex = BP[k]; 
    }
    #pragma omp parallel for    
    for (int k=0; k<dims[0]; k++) { 
       normBP[k] = BP[k] / bpex;
    }
    #pragma omp parallel for
    for (int k=0; k<dims[0]/2; k++) {
        halfBP = fabs(normBP[k] - 1.0 / pow(exp(1.0), 2.0));
        if (k == 0)
            bpex = halfBP;
        else if (halfBP < bpex) {
            bpex = halfBP;
            posi1 = k;
        }
    }
    #pragma omp parallel for 
    for (int k=dims[0]/2; k<dims[0]; k++) {
        halfBP = fabs(normBP[k] - 1.0/pow(exp(1.0), 2.0));
        if (k == dims[0]/2)
            bpex = halfBP;
        else if (halfBP < bpex) {
            bpex = halfBP;
            posi2 = k;
        }
    }
    
    *vald1oes = -r[posi1] + r[posi2];
    y[0] = r[posi1];
    y[1] = r[posi2];
    y[2] = BP[posi1];
    y[3] = BP[posi2];
}


void createPlans(int readwis, int *dims, const char* cpath, fftw_plan *pR1fft, fftw_plan *pR1ifft, fftw_plan *pR2fft, fftw_plan *pR2ifft, fftw_plan *pTfft, fftw_plan *pTifft) {

    #if USEGPU == 1

    /* direction is specified when the plan is executed */

    int idist = dims[0] * dims[1];
    int odist = dims[0] * dims[1];
    int istride = 1;
    int ostride = 1;
    int batch_size = dims[1] * dims[2];

    cufftPlanMany(pR1fft, 1, &dims[0], &dims[0], istride, idist, &dims[0], ostride, odist, CUFFT_Z2Z, batch_size);
        
    idist = dims[1];
    odist = dims[1];
    istride = dims[0];
    ostride = dims[0];
    batch_size = dims[0] * dims[2];
    
    cufftPlanMany(pR2fft, 1, &dims[0], &dims[1], istride, idist, &dims[1], ostride, odist, CUFFT_Z2Z, batch_size);

    idist = 1;
    odist = 1;
    istride = dims[0] * dims[1];
    ostride = dims[0] * dims[1];
    batch_size = dims[0] * dims[1];

    cufftPlanMany(pTfft, 1, &dims[0], &dims[2], istride, idist, &dims[2], ostride, odist, CUFFT_Z2Z, batch_size);
    
    #else

    /* Reading in wisdom */
    
    int len = 80;
    char wispath[len];
    snprintf(wispath, len, "%s%d", cpath, dims[0]);

    int savewis = 0;
    if (readwis) {
        if (!fftw_import_wisdom_from_filename(wispath)) {
            printf("Importing wisdom failed, creating new one to save.\n");
            savewis = 1;
        }
    }

    fftw_iodim64 *dim=malloc(1 * sizeof(fftw_iodim64));
    if (dim == NULL){fprintf(stderr,"malloc failed\n");exit(1);}
    fftw_iodim64 *howmany_dims=malloc(2 * sizeof(fftw_iodim64));
    if (howmany_dims == NULL){fprintf(stderr,"malloc failed\n"); exit(1);}

    cufftDoubleComplex *buffer;
    buffer  = (cufftDoubleComplex *)malloc(dims[0] * dims[1] * dims[2] * sizeof(cufftDoubleComplex));

    int howmany_rank;
    dim[0].n = dims[0]; // An array of size rank, so always 1 in this case. This one is along dim0
    dim[0].is = 1;
    dim[0].os = 1;
    howmany_rank = 2;
    howmany_dims[0].n = dims[1];
    howmany_dims[0].is = dims[0];
    howmany_dims[0].os = dims[0];
    howmany_dims[1].n = dims[2];
    howmany_dims[1].is = dims[1] * dims[0];
    howmany_dims[1].os= dims[1] * dims[0];

    *pR1fft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_FORWARD, FFTW_MEASURE);
    *pR1ifft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_BACKWARD, FFTW_MEASURE);
   
    dim[0].n = dims[1];
    dim[0].is =  dims[0];
    dim[0].os =  dims[0];
    howmany_rank = 2;
    howmany_dims[0].n = dims[0];
    howmany_dims[0].is = 1;
    howmany_dims[0].os= 1;
    howmany_dims[1].n = dims[2];
    howmany_dims[1].is = dims[1] * dims[0];
    howmany_dims[1].os= dims[1] * dims[0];

    *pR2fft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_FORWARD, FFTW_MEASURE);
    *pR2ifft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_BACKWARD, FFTW_MEASURE);
   
    dim[0].n = dims[2];
    dim[0].is =  dims[0] * dims[1];
    dim[0].os =  dims[0] * dims[1];
    howmany_rank = 2;
    howmany_dims[0].n = dims[0];
    howmany_dims[0].is = 1;
    howmany_dims[0].os = 1;
    howmany_dims[1].n = dims[1];
    howmany_dims[1].is = dims[0];
    howmany_dims[1].os = dims[0];

    *pTfft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_FORWARD, FFTW_MEASURE);
    *pTifft = fftw_plan_guru64_dft(1, dim, howmany_rank, howmany_dims, buffer, buffer, FFTW_BACKWARD, FFTW_MEASURE);

    free(buffer);
    free(dim);
    free(howmany_dims);

    if ((savewis == 1) || (readwis == 0)) {
        fftw_export_wisdom_to_filename(wispath);
    }
    #endif
}


void setStep(int *dims, double *dzz, cufftDoubleComplex *C1, cufftDoubleComplex *C2, double def_err, double dzmin, double *y1, double *y2, int *bad, double *err) {
   
    double sum1, sum2, err_fact, ujdz;
    
    *bad = 0;
    sum1 = 0;
    sum2 = 0;

    //#pragma omp parallel for default(shared) reduction(+:sum1, sum2)
    for (int i = 0; i<dims[0]*dims[1]*dims[2]; i++) { 
        cufftDoubleComplex CC;
        CC.x = C2[i].x - C1[i].x;
        CC.y = C2[i].y - C1[i].y;
        sum1 += pow(cuCabs(CC), 2); 
        sum2 += pow(cuCabs(C2[i]), 2); 
    } 
    
    *err = sqrt(sum1)/sqrt(sum2);
    printf("With step size: %.4f, the local error is: %.8g\nThe defined error is:%.8g\n ", *dzz, *err, def_err);

    err_fact = pow(2.0,(1.0/3.0)); //Split-step error
    if (*err > 2.0*def_err) {
        // Decrease step size and calc new solution
        ujdz = *dzz/2.0;
        if (ujdz > dzmin) {
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
    else if (*err > def_err && *err < 2.0 * def_err) {
        // Decrease step but don't recalc
        ujdz = *dzz/err_fact;
        if (ujdz > dzmin) {
            *dzz = ujdz;
            printf("Step decreased, no recalc. \n");
        }
        else {
            // must accept the step as it is too small already
            printf("Step accepted, would be too small otherwise\n");
        }
    }
    else if (*err > 0.5 * def_err && *err <= def_err) {
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
}


void sendmpi(cufftDoubleComplex *A, MPI_Datatype datatype, int *dims, int rankto) {
    
    /* approx 16 MB chunks */
    int div = 33554432 / 2;
    /* how many chunks */
    int nchunk = dims[0] * dims[1] * dims[2] / div;
    if (nchunk < 1) {
        /* it fits in one send */
        MPI_Send(A, dims[0] * dims[1] * dims[2], datatype, rankto, 0, MPI_COMM_WORLD);
    }
    else {
        for (int i=0; i<nchunk; ++i) {
            MPI_Send(A + i * div, dims[0] * dims[1] * dims[2] / nchunk, datatype, rankto, i, MPI_COMM_WORLD);
        }
    }
}


void recmpi(cufftDoubleComplex *A, MPI_Datatype datatype, int *dims, int rankfrom) {

    int div, nchunk;
    /* approx 16 MB chunks */
    div = 33554432/2;
    /* how many chunks */
    nchunk = dims[0]*dims[1]*dims[2]/div;
    if (nchunk < 1) {
        MPI_Recv(A, dims[0]*dims[1]*dims[2], datatype, rankfrom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else {
        for (int i=0; i<nchunk; i++) {
            MPI_Recv(A + i * div, dims[0] * dims[1] * dims[2]/nchunk, datatype, rankfrom, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}
