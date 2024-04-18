#include "propagation.h"

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