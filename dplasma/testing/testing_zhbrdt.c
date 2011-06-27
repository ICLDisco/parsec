/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

/* Including the bulge chassing */
#define FADDS_ZHBRDT(__n) (-1)
#define FMULS_ZHBRDT(__n) (-1)

static int check_solution(int, double*, double*, double);

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
    PASTE_CODE_IPARAM_LOCALS(iparam);
    if(P != 1) fprintf(stderr, "!!! This algorithm works on a band 1D matrix. The value of P=%d has been overriden, the actual grid is %dx%d\n", P, 1, nodes);
        
    PASTE_CODE_FLOPS_COUNT(FADDS_ZHBRDT, FMULS_ZHBRDT, ((DagDouble_t)N));
    PLASMA_Init(1);

    /*
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
         sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
         nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
         N, N, P, MatrixLower))
    */

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
                               two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                                      nodes, cores, rank, MB+1, NB+2, MB+1, (NB+2)*NT, 0, 0, 
                                                      MB+1, (NB+2)*NT, 1, SNB, 1 /* 1D cyclic */ ));

    generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescA, 100);


    PASTE_CODE_ENQUEUE_KERNEL(dague, zhbrdt, 
         ((tiled_matrix_desc_t*)&ddescA));

    PASTE_CODE_PROGRESS_KERNEL(dague, zhbrdt);

    if( check ) {
        printf( "No check implemented yet.\n" );

        /* Regenerate A, distributed so that the random generators are doing
         * the same things */
        PASTE_CODE_ALLOCATE_MATRIX(ddescAcpy, 1, 
                two_dim_block_cyclic, (&ddescAcpy, matrix_ComplexDouble, 
                    nodes, cores, rank, MB+1, NB+2, MB+1, (NB+2)*NT, 
                    0, 0, MB+1, (NB+2)*NT, 1, SNB, 1));
        generate_tiled_random_mat((tiled_matrix_desc_t*) &ddescAcpy, 100);
        /* Gather Acpy on rank 0 */
        PASTE_CODE_ALLOCATE_MATRIX(ddescLAcpy, 1, 
                two_dim_block_cyclic, (&ddescLAcpy, matrix_ComplexDouble, 
                    1, cores, rank, MB+1, NB+2, MB+1, (NB+2)*NT, 
                    0, 0, MB+1, (NB+2)*NT, 1, 1, 1));

        /* Gather A diagonal and subdiagonal on rank 0 */
        PASTE_CODE_ALLOCATE_MATRIX(ddescLA, 1, 
                two_dim_block_cyclic, (&ddescLA, matrix_ComplexDouble, 
                    1, cores, rank, 2, NB, 2, NB*NT, 
                    0, 0, 2, NB*NT, 1, 1, 1));
        if(rank == 0) {
            for(int t = 0; t < NT; t++)
            {
                int rsrc = ddescA.super.super.rank_of(0,t);
                if(rsrc == 0)
                {
                    PLASMA_Complex64_t* datain = ddescA.super.super.data_of(0,t);
                    PLASMA_Complex64_t* dataout = ddescLA.super.super.data_of(0,t);
                    for(int n = 0; n < NB; n++) for(int m = 0; m < 2; m++) 
                    {
                        dataout[m+n*2] = datain[m+n*(MB+1)];
                    }
                }
                else
                {
                    PLASMA_Complex64_t* dataout = ddescLA.super.super.data_of(0,t);
                    MPI_Recv(dataout, 2*NB, MPI_DOUBLE_COMPLEX, rsrc, t, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
        else
        {
            MPI_Datatype bidiagband_dtt; 
            MPI_Type_vector(NB, 2, MB+1, MPI_DOUBLE_COMPLEX, &bidiagband_dtt); 

            for(int t = 0; t < NT; t++) {
                if(ddescA.super.super.rank_of(0,t) == rank)
                {
                    PLASMA_Complex64_t* datain = ddescA.super.super.data_of(0,t);
                    MPI_Send(datain, 1, bidiagband_dtt, 0, t, MPI_COMM_WORLD);
                }
            }
        }
    }
    dplasma_zhbrdt_Destruct( DAGUE_zhbrdt );
    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    
    cleanup_dague(dague, iparam);
        
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
