/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include <plasma.h>
#include "common.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"

/* Including the bulge chassing */
#define FADDS_ZHBRDT(__n) (-1)
#define FMULS_ZHBRDT(__n) (-1)

static int check_orthogonality(int, int, int, PLASMA_Complex64_t*, double);
static int check_reduction(int, int, int, PLASMA_Complex64_t*, PLASMA_Complex64_t*, int, PLASMA_Complex64_t*, double);
static int check_solution(int, double*, double*, double);
static int check_solution2(int, double*, double*, double);

int main(int argc, char *argv[])
{
    dague_context_t *dague;
    int iparam[IPARAM_SIZEOF];

     /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 48, 144, 144);
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#endif

    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS_COUNT(FADDS_ZHBRDT, FMULS_ZHBRDT, ((DagDouble_t)N))

    LDA = max(M, LDA);
    LDB = max( LDB, N );
    SMB = 1; SNB = 1;

    PLASMA_Init(1);

    /*
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
         sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
         nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
         N, N, P, MatrixLower))
    */

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
                               two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                                      nodes, cores, rank, MB, NB, MB, NB*NT, 0, 0, 
                                                      NB, NB*NT, 1, SNB, 1 /* 1D cyclic */ ));

    generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescA, 100);

    PLASMA_Complex64_t *A2 = (PLASMA_Complex64_t *)malloc(LDA*N*sizeof(PLASMA_Complex64_t));
    double *W1             = (double *)malloc(N*sizeof(double));
    double *W2             = (double *)malloc(N*sizeof(double));

    if( check ) {
        printf( "No check implemented yet.\n" );
    }

    PASTE_CODE_ENQUEUE_KERNEL(dague, zhbrdt, 
         ((tiled_matrix_desc_t*)&ddescA));

    PASTE_CODE_PROGRESS_KERNEL(dague, zhbrdt);

    free(A2); free(W1); free(W2);
    dplasma_zhbrdt_Destruct( DAGUE_zhbrdt );

    dague_data_free(ddescA.mat);
    
    cleanup_dague(dague);
        
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
        
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
    double maxel = fabs(fabs(E1[0])-fabs(E2[0]));
    double maxeig = fmax(fabs(E1[0]), fabs(E2[0]));
    for (i = 1; i < N; i++){
        Residual[i] = fabs(fabs(E1[i])-fabs(E2[i]));
        maxtmp      = fmax(fabs(E1[i]), fabs(E2[i]));
        maxeig      = fmax(maxtmp, maxeig);
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
    if (isnan(maxel / eps) || isinf(maxel / eps) || ((maxel / (maxeig*eps)) > 1000.0) ) {
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
