/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "dague_internal.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/diag_band_to_rect.h"
#include "dplasma/lib/dplasmatypes.h"

/* Including the bulge chassing */
#define FADDS_ZHEEV(__n) (((__n) * (-8.0 / 3.0 + (__n) * (1.0 + 2.0 / 3.0 * (__n)))) - 4.0)
#define FMULS_ZHEEV(__n) (((__n) * (-1.0 / 6.0 + (__n) * (5.0 / 2.0 + 2.0 / 3.0 * (__n)))) - 15.0)

static inline double dplasma_fmax( double a, double b){
    return ( a > b ) ? a : b;
}

static int check_solution(int, double*, double*, double);

int main(int argc, char *argv[])
{
    int i, j;
    dague_context_t *dague;
    int iparam[IPARAM_SIZEOF];
    PLASMA_enum uplo = PlasmaLower;
    PLASMA_desc *plasmaDescA;

     /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    //iparam_default_ibnbmb(iparam, 48, 144, 144);
    iparam_default_ibnbmb(iparam, 4, 4, 4);
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#endif

    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    PASTE_CODE_FLOPS_COUNT(FADDS_ZHEEV, FMULS_ZHEEV, ((DagDouble_t)N));

    PLASMA_Init(1);
    PLASMA_Disable(PLASMA_AUTOTUNING);
    PLASMA_Set(PLASMA_TILE_SIZE, MB);

    PLASMA_Complex64_t *A2 = (PLASMA_Complex64_t *)malloc(LDA*N*sizeof(PLASMA_Complex64_t));
    double *W1             = (double *)malloc(N*sizeof(double));
    double *W2             = (double *)malloc(N*sizeof(double));
    double *D             = (double *)malloc(N*sizeof(double));
    double *E             = (double *)malloc(N*sizeof(double));
    int INFO;

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
         two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
         nodes, cores, rank, MB, NB, LDA, N, 0, 0,
         N, N, 1, 1, P))
    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1,
         two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, matrix_Tile,
         nodes, cores, rank, IB, NB, MT*IB, N, 0, 0,
         MT*IB, N, 1, 1, P))
/*
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
         sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
         nodes, cores, rank, MB, NB, LDA, N, 0, 0,
         N, N, P, uplo))
    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1,
         sym_two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble,
         nodes, cores, rank, IB, NB, MT*IB, N, 0, 0,
         MT*IB, N, P, uplo))
*/
    PASTE_CODE_ALLOCATE_MATRIX(ddescBAND, 1,
        two_dim_block_cyclic, (&ddescBAND, matrix_ComplexDouble, matrix_Tile,
        nodes, cores, rank, MB+1, NB+2, MB+1, (NB+2)*NT, 0, 0,
        MB+1, (NB+2)*NT, 1, SNB, 1 /* 1D cyclic */ ));

    PLASMA_Desc_Create(&plasmaDescA, ddescA.mat, PlasmaComplexDouble,
         ddescA.super.mb, ddescA.super.nb, ddescA.super.bsiz,
         ddescA.super.lm, ddescA.super.ln, ddescA.super.i, ddescA.super.j,
         ddescA.super.m, ddescA.super.n);

    dplasma_zplghe( dague, (double)N, uplo, (tiled_matrix_desc_t *)&ddescA, 1358);

    if( check ) {
        PLASMA_Tile_to_Lapack(plasmaDescA, (void*)A2, LDA);

        /*
        */
        printf("A2 avant\n");
        for (i = 0; i < N; i++){
            for (j = 0; j < N; j++) {
                //printf("%f+%fi ",
                    //creal(A2[N*j+i]),
                    //cimag(A2[N*j+i]));
#if defined(PRECISION_d) || defined(PRECISION_s)
                printf("%f ", A2[LDA*j+i] );
#else
                printf("(%f, %f)", creal(A2[LDA*j+i]), cimag(A2[LDA*j+i]));
#endif
            }
            printf("\n");
        }

        LAPACKE_zheev( LAPACK_COL_MAJOR,
               lapack_const(PlasmaNoVec), lapack_const(uplo),
               N, A2, LDA, W1);

        printf("Eigenvalues original\n");
        for(i = 0; i < N; i++){
            printf("%f\n", W1[i]);
        }
        printf("\n");
    }

    PASTE_CODE_ENQUEUE_KERNEL(dague, zherbt,
         (uplo, IB, (tiled_matrix_desc_t*)&ddescA, (tiled_matrix_desc_t*)&ddescT));
    PASTE_CODE_PROGRESS_KERNEL(dague, zherbt);

    SYNC_TIME_START();
    dague_diag_band_to_rect_object_t* DAGUE_diag_band_to_rect = dague_diag_band_to_rect_new((sym_two_dim_block_cyclic_t*)&ddescA, &ddescBAND,
            MT, NT, MB, NB, sizeof(matrix_ComplexDouble));
    dague_arena_t* arena = DAGUE_diag_band_to_rect->arenas[DAGUE_diag_band_to_rect_DEFAULT_ARENA];
    dplasma_add2arena_tile(arena,
                           MB*NB*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, MB);
    dague_enqueue(dague, (dague_object_t*)DAGUE_diag_band_to_rect);
    dague_progress(dague);
    SYNC_TIME_PRINT(rank, ( "diag_band_to_rect N= %d NB = %d : %f s\n", N, NB, sync_time_elapsed));

    PASTE_CODE_ENQUEUE_KERNEL(dague, zhbrdt, ((tiled_matrix_desc_t*)&ddescBAND));
    PASTE_CODE_PROGRESS_KERNEL(dague, zhbrdt)

    if( check ) {
        PLASMA_Tile_to_Lapack(plasmaDescA, (void*)A2, LDA);
#if 0
        {
          int k, sizearena = (NB+1)*(NB+2);
          /* store resulting diag and lower diag D and E*/
          for (k=0;k<NT-1;k++) {
            for (j=0;j<NB;j++) {
              D[(k*NB)+j] = ddescBAND.mat[(k*sizearena)+ LDA*j];
              E[(k*NB)+j] = ddescBAND.mat[(k*sizearena)+ LDA*j+1];
            }
          }
          k=NT-1;
          for (j=0;j<NB-1;j++) {
            D[(k*NB)+j] = ddescBAND.mat[(k*sizearena)+ LDA*j];
            E[(k*NB)+j] = ddescBAND.mat[(k*sizearena)+ LDA*j+1];
          }
          D[(k*NB)+(NB-1)] = ddescBAND.mat[(k*sizearena)+ LDA*(NB-1)];
        }
#endif

        /* call eigensolver */
        dsterf_( &N, D, E, &INFO);

        /*
        */
        printf("A2 apres\n");
        for (i = 0; i < N; i++){
            for (j = 0; j < N; j++) {
#if defined(PRECISION_d) || defined(PRECISION_s)
                printf("%f ", A2[LDA*j+i] );
#else
                printf("(%f, %f)", creal(A2[LDA*j+i]), cimag(A2[LDA*j+i]));
#endif
            }
            printf("\n");
        }

        for (j = 0; j < N; j++)
            for (i = j+2; i < N; i++)
                A2[LDA*j+i]=0.0;

        LAPACKE_zheev( LAPACK_COL_MAJOR,
               lapack_const(PlasmaNoVec), lapack_const(uplo),
               N, A2, LDA, W2);


        printf("Eigenvalues computed\n");
        for (i = 0; i < N; i++){
            printf("%f \n", W2[i]);
        }
        printf("\n");

        double eps = LAPACKE_dlamch_work('e');
        printf("\n");
        printf("------ TESTS FOR PLASMA ZHEEV ROUTINE -------  \n");
        printf("        Size of the Matrix %d by %d\n", N, N);
        printf("\n");
        printf(" The matrix A is randomly generated for each test.\n");
        printf("============\n");
        printf(" The relative machine precision (eps) is to be %e \n",eps);
        printf(" Computational tests pass if scaled residuals are less than 60.\n");

        /* Check the eigen solutions */
        int info_solution = check_solution(N, W1, W2, eps);

        if (info_solution == 0) {
            printf("***************************************************\n");
            printf(" ---- TESTING ZHEEV ..................... PASSED !\n");
            printf("***************************************************\n");
        }
        else {
            printf("************************************************\n");
            printf(" - TESTING ZHEEV ..................... FAILED !\n");
            printf("************************************************\n");
        }
    }

    dplasma_zherbt_Destruct( DAGUE_zherbt );
    DAGUE_INTERNAL_OBJECT_DESTRUCT( DAGUE_diag_band_to_rect );
    dplasma_zhbrdt_Destruct( DAGUE_zhbrdt );

    cleanup_dague(dague, iparam);

    free(A2); free(W1); free(W2); free(D); free(E);
    dague_data_free(ddescBAND.mat);
    dague_data_free(ddescA.mat);
    dague_data_free(ddescT.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescBAND);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescT);

    PLASMA_Finalize();

    return EXIT_SUCCESS;
}



/*--------------------------------------------------------------
 * Check the solution
 */

static int check_solution(int N, double *E1, double *E2, double eps)
{
    int info_solution, i;
    double *Residual = (double *)malloc(N*sizeof(double));
    double maxtmp;
    double maxel  = fabs(fabs(E1[0])-fabs(E2[0]));
    double maxeig = dplasma_fmax(fabs(E1[0]), fabs(E2[0]));
    for (i = 1; i < N; i++){
        Residual[i] = fabs(fabs(E1[i])-fabs(E2[i]));
        maxtmp      = dplasma_fmax(fabs(E1[i]), fabs(E2[i]));
        maxeig      = dplasma_fmax(maxtmp, maxeig);
        //printf("Residu: %f E1: %f E2: %f\n", Residual[i], E1[i], E2[i] );
        if (maxel < Residual[i])
           maxel =  Residual[i];
    }

    //printf("maxel: %.16f maxeig: %.16f \n", maxel, maxeig );

    printf(" ======================================================\n");
    printf(" | D -  eigcomputed | / (|D| ulp)      : %15.3E \n",  maxel/(maxeig*eps) );
    printf(" ======================================================\n");


    printf("============\n");
    printf("Checking the eigenvalues of A\n");
    if (isnan(maxel / eps) || isinf(maxel / eps) || ((maxel / (maxeig*eps)) > 100.0) ) {
        //printf("isnan: %d %f %e\n", isnan(maxel / eps), maxel, eps );
        //printf("isinf: %d %f %e\n", isinf(maxel / eps), maxel, eps );
        printf("-- The eigenvalues are suspicious ! \n");
        info_solution = 1;
    }
    else{
        printf("-- The eigenvalues are CORRECT ! \n");
        info_solution = 0;
    }

    free(Residual);
    return info_solution;
}

