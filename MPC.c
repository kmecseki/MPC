#define _USE_MATH_DEFINES
#include <complex.h>
#include <stdlib.h>

#include <math.h>
#include <string.h>
#include <time.h>

#include <getopt.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "constants.h"
#include "settings.h"
#include "utils.h"
#include "kernels.h"
#include "propagation.h"
#include "plasma.h"



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
