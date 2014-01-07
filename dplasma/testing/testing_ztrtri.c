/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

static int check_solution( dague_context_t *dague, int loud,
                           PLASMA_enum uplo, PLASMA_enum diag, int N,
                           tiled_matrix_desc_t *A,
                           tiled_matrix_desc_t *Ainv );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;
    int Aseed = 3872;

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
#if defined(HAVE_CUDA) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    /* initializing matrix structure */
    int Am = dplasma_imax(M, N);

    LDA = dplasma_imax(LDA, Am);
    LDC = dplasma_imax(LDC, M);
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, Am, 0, 0,
                               Am, Am, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescAinv, check,
        two_dim_block_cyclic, (&ddescAinv, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, Am, 0, 0,
                               Am, Am, SMB, SNB, P));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    /* Generate matrix A with diagonal dominance to keep stability during computation */
    dplasma_zplrnt( dague, 1, (tiled_matrix_desc_t *)&ddescA, Aseed);
    /* Scale down the full matrix to keep stability in diag = PlasmaUnit case */
    dplasma_zlascal( dague, PlasmaUpperLower,
                     1. / (dague_complex64_t)Am,
                     (tiled_matrix_desc_t *)&ddescA );
    if(loud > 2) printf("Done\n");

    if(!check)
    {
        PLASMA_enum uplo  = PlasmaLower;
        PLASMA_enum diag  = PlasmaUnit;
        int info = 0;

        PASTE_CODE_FLOPS(FLOPS_ZTRTRI, ((DagDouble_t)Am));

        /* Create DAGuE */
        PASTE_CODE_ENQUEUE_KERNEL(dague, ztrtri,
                                  (uplo, diag, (tiled_matrix_desc_t *)&ddescA, &info));

        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(dague, ztrtri);

        dplasma_ztrtri_Destruct( DAGUE_ztrtri );
    }
    else
    {
        int u, d, info;
        int info_solution;

        for (u=0; u<2; u++) {
            for (d=0; d<2; d++) {
                if ( rank == 0 ) {
                    printf("***************************************************\n");
                    printf(" ----- TESTING ZTRTRI (%s, %s) -------- \n",
                           uplostr[u], diagstr[d]);
                }

                /* matrix generation */
                printf("Generate matrices ... ");
                dplasma_zlacpy( dague, PlasmaUpperLower,
                                (tiled_matrix_desc_t *)&ddescA,
                                (tiled_matrix_desc_t *)&ddescAinv );
                printf("Done\n");

                /* Compute */
                printf("Compute ... ... ");
                info = dplasma_ztrtri(dague, uplo[u], diag[d],
                               (tiled_matrix_desc_t *)&ddescAinv);
                printf("Done\n");

                /* Check the solution */
                if (info != 0) {
                    info_solution = 1;

                } else {
                    info_solution = check_solution(dague, rank == 0 ? loud : 0,
                                                   uplo[u], diag[d], Am,
                                                   (tiled_matrix_desc_t*)&ddescA,
                                                   (tiled_matrix_desc_t*)&ddescAinv);
                }
                if ( rank == 0 ) {
                    if (info_solution == 0) {
                        printf(" ---- TESTING ZTRTRI (%s, %s) ...... PASSED !\n",
                               uplostr[u], diagstr[d]);
                    }
                    else {
                        printf(" ---- TESTING ZTRTRI (%s, %s) ... FAILED !\n",
                               uplostr[u], diagstr[d]);
                        ret |= 1;
                    }
                    printf("***************************************************\n");
                }
            }
        }

        dague_data_free(ddescAinv.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescAinv);
    }

    cleanup_dague(dague, iparam);

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);

    return ret;
}


/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution( dague_context_t *dague, int loud,
                           PLASMA_enum uplo, PLASMA_enum diag, int N,
                           tiled_matrix_desc_t *A,
                           tiled_matrix_desc_t *Ainv )
{
    two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A;
    int info_solution;
    double Anorm, Ainvnorm, Rnorm;
    double Rcond;
    double eps, result;

    eps = LAPACKE_dlamch_work('e');

    PASTE_CODE_ALLOCATE_MATRIX(Id, 1,
        two_dim_block_cyclic, (&Id, matrix_ComplexDouble, matrix_Tile,
                               A->super.nodes, A->super.cores, twodA->grid.rank,
                               A->mb, A->nb, N, N, 0, 0,
                               N, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    PASTE_CODE_ALLOCATE_MATRIX(A0, 1,
        two_dim_block_cyclic, (&A0, matrix_ComplexDouble, matrix_Tile,
                               A->super.nodes, A->super.cores, twodA->grid.rank,
                               A->mb, A->nb, N, N, 0, 0,
                               N, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&Id);
    if ( diag == PlasmaNonUnit ) {
        dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&A0);
        dplasma_zlacpy( dague, uplo, Ainv, (tiled_matrix_desc_t *)&A0 );
    } else {
        PLASMA_enum nuplo = (uplo == PlasmaLower) ? PlasmaUpper : PlasmaLower ;

        dplasma_zlacpy( dague, uplo,  Ainv,   (tiled_matrix_desc_t *)&A0 );
        dplasma_zlaset( dague, nuplo, 0., 1., (tiled_matrix_desc_t *)&A0 );
    }
    dplasma_ztrmm(dague, PlasmaLeft, uplo, PlasmaNoTrans, diag,
                  1., A, (tiled_matrix_desc_t *)&A0 );
    dplasma_zgeadd( dague, PlasmaUpperLower, -1.0,
                    (tiled_matrix_desc_t*)&A0,
                    (tiled_matrix_desc_t*)&Id );

    Anorm    = dplasma_zlantr( dague, PlasmaOneNorm, uplo, diag, A );
    Ainvnorm = dplasma_zlantr( dague, PlasmaOneNorm, uplo, diag, Ainv );
    Rnorm    = dplasma_zlantr( dague, PlasmaOneNorm, uplo, PlasmaNonUnit, (tiled_matrix_desc_t*)&Id );

    Rcond  = ( 1. / Anorm ) / Ainvnorm;
    result = (Rnorm * Rcond) / (eps * N);

    if ( loud > 2 ) {
        printf("  ||A||_one = %e, ||A^(-1)||_one = %e, ||I - A * A^(-1)||_one = %e, cond = %e, result = %e\n",
               Anorm, Ainvnorm, Rnorm, Rcond, result);
    }

    if ( isinf(Ainvnorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
        info_solution = 1;
    }
    else {
        info_solution = 0;
    }

    dague_data_free(Id.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&Id);
    dague_data_free(A0.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&A0);

    return info_solution;
}
