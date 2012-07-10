///////////////////////////////////////////////////////////////////////////////////////////////////

// size of work for a thread
#define THR_M ( BLK_M / DIM_X )
#define THR_N ( BLK_N / DIM_Y )
 
///////////////////////////////////////////////////////////////////////////////////////////////////

#if   (version == trans_nn)
#define kernel_name GENERATE_SM_VERSION_KERNEL_NAME(nn)
#elif (version == trans_nt)
#define TRANS_B
#define kernel_name GENERATE_SM_VERSION_KERNEL_NAME(nt)
#elif (version == trans_nc)
#define TRANS_B 
#define CONJ_B
#define kernel_name GENERATE_SM_VERSION_KERNEL_NAME(nc)
#elif (version == trans_tn)
#define TRANS_A
#define kernel_name GENERATE_SM_VERSION_KERNEL_NAME(tn)
#elif (version == trans_tt)
#define TRANS_A
#define TRANS_B 
#define kernel_name GENERATE_SM_VERSION_KERNEL_NAME(tt)
#elif (version == trans_tc)
#define TRANS_A
#define TRANS_B
#define CONJ_B
#define kernel_name GENERATE_SM_VERSION_KERNEL_NAME(tc)
#elif (version == trans_cn)
#define TRANS_A
#define CONJ_A
#define kernel_name GENERATE_SM_VERSION_KERNEL_NAME(cn)
#elif (version == trans_ct)
#define TRANS_A
#define CONJ_A
#define TRANS_B
#define kernel_name GENERATE_SM_VERSION_KERNEL_NAME(ct)
#elif (version == trans_cc)
#define TRANS_A
#define CONJ_A
#define TRANS_B
#define CONJ_B
#define kernel_name GENERATE_SM_VERSION_KERNEL_NAME(cc)
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" __global__
void     kernel_name (int M, int N, int K,
                      FloatingPoint_t alpha, const FloatingPoint_t *A, int LDA,
                                             const FloatingPoint_t *B, int LDB, 
                      FloatingPoint_t beta,        FloatingPoint_t *C, int LDC,
                      int offsetA, int offsetB)
{
    int idx = threadIdx.x;  // thread's m dimension
    int idy = threadIdx.y;  // thread's n dimension

    int idt = DIM_X * idy + idx;    // thread's global number

    int idxA = idt % DIM_XA;    // idx within A
    int idyA = idt / DIM_XA;    // idy within A

    int idxB = idt % DIM_XB;    // idx within B
    int idyB = idt / DIM_XB;    // idy within B

    int blx = blockIdx.x;   // block's m dimension
    int bly = blockIdx.y;   // block's n dimension

    __shared__ FloatingPoint_t sA[BLK_K][BLK_M+1];      // +1 only required if A is transposed
    __shared__ FloatingPoint_t sB[BLK_N][BLK_K+1];      // +1 always required

    // Registers for the innermost loop
    FloatingPoint_t rC[THR_N][THR_M];
    FloatingPoint_t rA[THR_M];
    FloatingPoint_t rB[THR_N];

    // Registers for the dev->shmem copy
    #ifdef TRANS_A
        FloatingPoint_t ra[BLK_M/DIM_YA][BLK_K/DIM_XA];
    #else
        FloatingPoint_t ra[BLK_K/DIM_YA][BLK_M/DIM_XA];
    #endif
    #ifdef TRANS_B
        FloatingPoint_t rb[BLK_K/DIM_YB][BLK_N/DIM_XB];
    #else
        FloatingPoint_t rb[BLK_N/DIM_YB][BLK_K/DIM_XB];
    #endif

      #ifdef TEXTURE_1D
        #ifdef TRANS_A
            int coord_A = offsetA + blx*BLK_M*LDA + idyA*LDA+idxA;
        #else
            int coord_A = offsetA + blx*BLK_M     + idyA*LDA+idxA;
        #endif
        #ifdef TRANS_B
            int coord_B = offsetB + bly*BLK_N     + idyB*LDB+idxB;
        #else
            int coord_B = offsetB + bly*BLK_N*LDB + idyB*LDB+idxB;
        #endif
      #else
        #ifdef TRANS_A
          FloatingPoint_t *offs_dA = A + blx*BLK_M*LDA + idyA*LDA+idxA;
        #else
          FloatingPoint_t *offs_dA = A + blx*BLK_M     + idyA*LDA+idxA;
        #endif
        #ifdef TRANS_B
          FloatingPoint_t *offs_dB = B + bly*BLK_N     + idyB*LDB+idxB;
        #else
          FloatingPoint_t *offs_dB = B + bly*BLK_N*LDB + idyB*LDB+idxB;
        #endif
      #endif

    int m, n, k, kk;

    // Zero C
    #pragma unroll
    for (n = 0; n < THR_N; n++)
      #pragma unroll
      for (m = 0; m < THR_M; m++)
        rC[n][m] = make_FloatingPoint(0.0, 0.0);

    // Load A dev->shmem
    #ifdef TRANS_A
        #pragma unroll
        for (n = 0; n < BLK_M; n += DIM_YA)
          #pragma unroll
          for (m = 0; m < BLK_K; m += DIM_XA)
            sA[m+idxA][n+idyA] = fetch(A, m, n);
    #else
        #pragma unroll
        for (n = 0; n < BLK_K; n += DIM_YA)
          #pragma unroll
          for (m = 0; m < BLK_M; m += DIM_XA)
            sA[n+idyA][m+idxA] = fetch(A, m, n);
    #endif

    // Load B dev->shmem
    #ifdef TRANS_B
        #pragma unroll
        for (n = 0; n < BLK_K; n += DIM_YB)
          #pragma unroll
          for (m = 0; m < BLK_N; m += DIM_XB)
            sB[m+idxB][n+idyB] = fetch(B, m, n);
    #else
        #pragma unroll
        for (n = 0; n < BLK_N; n += DIM_YB)
          #pragma unroll
          for (m = 0; m < BLK_K; m += DIM_XB)
            sB[n+idyB][m+idxB] = fetch(B, m, n);
    #endif

    __syncthreads();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K)
    {
          #ifdef TEXTURE_1D
            #ifdef TRANS_A
                coord_A += BLK_K;
            #else
                coord_A += BLK_K*LDA;
            #endif
            #ifdef TRANS_B
                coord_B += BLK_K*LDB;
            #else
                coord_B += BLK_K;
            #endif
          #else
            #ifdef TRANS_A
                offs_dA += BLK_K;
            #else
                offs_dA += BLK_K*LDA;
            #endif
            #ifdef TRANS_B
                offs_dB += BLK_K*LDB;
            #else
                offs_dB += BLK_K;
            #endif
          #endif

        // Load A dev->regs
        #ifdef TRANS_A
            #pragma unroll
            for (n = 0; n < BLK_M/DIM_YA; n++)
              #pragma unroll
              for (m = 0; m < BLK_K/DIM_XA; m++)
                ra[n][m] = fetch(A, m*DIM_XA, n*DIM_YA);
        #else
            #pragma unroll
            for (n = 0; n < BLK_K/DIM_YA; n++)
              #pragma unroll
              for (m = 0; m < BLK_M/DIM_XA; m++)
                ra[n][m] = fetch(A, m*DIM_XA, n*DIM_YA);
        #endif

        // Load B dev->regs
        #ifdef TRANS_B
            #pragma unroll
            for (n = 0; n < BLK_K/DIM_YB; n++)
              #pragma unroll
              for (m = 0; m < BLK_N/DIM_XB; m++)
                rb[n][m] = fetch(B, m*DIM_XB, n*DIM_YB);
        #else
            #pragma unroll
            for (n = 0; n < BLK_N/DIM_YB; n++)
              #pragma unroll
              for (m = 0; m < BLK_K/DIM_XB; m++)
                rb[n][m] = fetch(B, m*DIM_XB, n*DIM_YB);
        #endif

        // Multiply
        #pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            // Load A shmem->regs
            #pragma unroll
            for (m = 0; m < THR_M; m++)
                rA[m] = sA[k][m*DIM_X+idx];

            // Load B shmem->regs
            #pragma unroll
            for (n = 0; n < THR_N; n++)
                rB[n] = sB[n*DIM_Y+idy][k];

            // Compute
            #pragma unroll
            for (n = 0; n < THR_N; n++)
                #pragma unroll
                for (m = 0; m < THR_M; m++)
                    #ifdef CONJ_A
                      #ifdef CONJ_B
                        fma(conj(rA[m]), conj(rB[n]), rC[n][m]);
                      #else
                        fma(conj(rA[m]), rB[n], rC[n][m]);
                      #endif
                    #else
                      #ifdef CONJ_B
                        fma(rA[m], conj(rB[n]), rC[n][m]);
                      #else
                        fma(rA[m], rB[n], rC[n][m]);
                      #endif
                    #endif
        }

        __syncthreads();

        // Load A regs->shmem
        #ifdef TRANS_A
            #pragma unroll
            for (n = 0; n < BLK_M/DIM_YA; n++)
              #pragma unroll
              for (m = 0; m < BLK_K/DIM_XA; m++)
                sA[m*DIM_XA+idxA][n*DIM_YA+idyA] = ra[n][m];
        #else
            #pragma unroll
            for (n = 0; n < BLK_K/DIM_YA; n++)
              #pragma unroll
              for (m = 0; m < BLK_M/DIM_XA; m++)
                sA[n*DIM_YA+idyA][m*DIM_XA+idxA] = ra[n][m];
            #endif

        // Load B regs->shmem
        #ifdef TRANS_B
            #pragma unroll
            for (n = 0; n < BLK_K/DIM_YB; n++)
              #pragma unroll
              for (m = 0; m < BLK_N/DIM_XB; m++)
                sB[m*DIM_XB+idxB][n*DIM_YB+idyB] = rb[n][m];
        #else
            #pragma unroll
            for (n = 0; n < BLK_N/DIM_YB; n++)
              #pragma unroll
              for (m = 0; m < BLK_K/DIM_XB; m++)
                sB[n*DIM_YB+idyB][m*DIM_XB+idxB] = rb[n][m];
        #endif

        __syncthreads();
    }

    // Multiply
    #pragma unroll
    for (k = 0; k < BLK_K; k++)
    {
        // Load A shmem->regs
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rA[m] = sA[k][m*DIM_X+idx];

        // Load B shmem->regs
        #pragma unroll
        for (n = 0; n < THR_N; n++)
            rB[n] = sB[n*DIM_Y+idy][k];

        // Compute
        #pragma unroll
        for (n = 0; n < THR_N; n++)
            #pragma unroll
            for (m = 0; m < THR_M; m++)
                #ifdef CONJ_A
                  #ifdef CONJ_B
                    fma(conj(rA[m]), conj(rB[n]), rC[n][m]);
                  #else
                    fma(conj(rA[m]), rB[n], rC[n][m]);
                  #endif
                #else
                  #ifdef CONJ_B
                    fma(rA[m], conj(rB[n]), rC[n][m]);
                  #else
                    fma(rA[m], rB[n], rC[n][m]);
                  #endif
                #endif
    }

    // Store C regs->dev
    #pragma unroll
    for (n = 0; n < THR_N; n++) {
        int coord_dCn = bly*BLK_N + n*DIM_Y+idy;
        #pragma unroll
        for (m = 0; m < THR_M; m++) {
            int coord_dCm = blx*BLK_M + m*DIM_X+idx;
            if (coord_dCm < M && coord_dCn < N) {
                int offsC = coord_dCn*LDC + coord_dCm;

                FloatingPoint_t &regC = rC[n][m];
                FloatingPoint_t &memC = C[offsC];

                memC = add(mul(alpha, regC), mul(beta, memC));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#undef TRANS_A
#undef TRANS_B
#undef CONJ_A
#undef CONJ_B

#undef BLK_M
#undef BLK_N
#undef BLK_K

/* #undef DIM_X */
/* #undef DIM_Y */

#undef DIM_XA
#undef DIM_YA 

#undef DIM_XB
#undef DIM_YB

#undef version

#undef THR_M
#undef THR_N

#undef kernel_name
