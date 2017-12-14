/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

static int check_solution( parsec_context_t *parsec, int loud,
                           PLASMA_enum uplo, PLASMA_enum diag, int N,
                           parsec_tiled_matrix_dc_t *A,
                           parsec_tiled_matrix_dc_t *Ainv );

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;
    int Aseed = 3872;

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
#if defined(PARSEC_HAVE_CUDA) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    /* initializing matrix structure */
    int Am = dplasma_imax(M, N);

    LDA = dplasma_imax(LDA, Am);
    LDC = dplasma_imax(LDC, M);
    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, Am, 0, 0,
                               Am, Am, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(dcAinv, check,
        two_dim_block_cyclic, (&dcAinv, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, Am, 0, 0,
                               Am, Am, SMB, SNB, P));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    /* Generate matrix A with diagonal dominance to keep stability during computation */
    dplasma_zplrnt( parsec, 1, (parsec_tiled_matrix_dc_t *)&dcA, Aseed);
    /* Scale down the full matrix to keep stability in diag = PlasmaUnit case */
    dplasma_zlascal( parsec, PlasmaUpperLower,
                     1. / (parsec_complex64_t)Am,
                     (parsec_tiled_matrix_dc_t *)&dcA );
    if(loud > 2) printf("Done\n");

    if(!check)
    {
        PLASMA_enum uplo  = PlasmaLower;
        PLASMA_enum diag  = PlasmaUnit;
        int info = 0;

        PASTE_CODE_FLOPS(FLOPS_ZTRTRI, ((DagDouble_t)Am));

        /* Create PaRSEC */
        PASTE_CODE_ENQUEUE_KERNEL(parsec, ztrtri,
                                  (uplo, diag, (parsec_tiled_matrix_dc_t *)&dcA, &info));

        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(parsec, ztrtri);

        dplasma_ztrtri_Destruct( PARSEC_ztrtri );
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
                dplasma_zlacpy( parsec, PlasmaUpperLower,
                                (parsec_tiled_matrix_dc_t *)&dcA,
                                (parsec_tiled_matrix_dc_t *)&dcAinv );
                printf("Done\n");

                /* Compute */
                printf("Compute ... ... ");
                info = dplasma_ztrtri(parsec, uplo[u], diag[d],
                               (parsec_tiled_matrix_dc_t *)&dcAinv);
                printf("Done\n");

                /* Check the solution */
                if (info != 0) {
                    info_solution = 1;

                } else {
                    info_solution = check_solution(parsec, rank == 0 ? loud : 0,
                                                   uplo[u], diag[d], Am,
                                                   (parsec_tiled_matrix_dc_t*)&dcA,
                                                   (parsec_tiled_matrix_dc_t*)&dcAinv);
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

        parsec_data_free(dcAinv.mat);
        parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcAinv);
    }

    cleanup_parsec(parsec, iparam);

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcA);

    return ret;
}


/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution( parsec_context_t *parsec, int loud,
                           PLASMA_enum uplo, PLASMA_enum diag, int N,
                           parsec_tiled_matrix_dc_t *A,
                           parsec_tiled_matrix_dc_t *Ainv )
{
    two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A;
    int info_solution = 1;
    double Anorm, Ainvnorm, Rnorm;
    double Rcond;
    double eps, result;

    eps = LAPACKE_dlamch_work('e');

    PASTE_CODE_ALLOCATE_MATRIX(Id, 1,
        two_dim_block_cyclic, (&Id, matrix_ComplexDouble, matrix_Tile,
                               A->super.nodes, twodA->grid.rank,
                               A->mb, A->nb, N, N, 0, 0,
                               N, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    PASTE_CODE_ALLOCATE_MATRIX(A0, 1,
        two_dim_block_cyclic, (&A0, matrix_ComplexDouble, matrix_Tile,
                               A->super.nodes, twodA->grid.rank,
                               A->mb, A->nb, N, N, 0, 0,
                               N, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (parsec_tiled_matrix_dc_t *)&Id);
    if ( diag == PlasmaNonUnit ) {
        dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (parsec_tiled_matrix_dc_t *)&A0);
        dplasma_zlacpy( parsec, uplo, Ainv, (parsec_tiled_matrix_dc_t *)&A0 );
    } else {
        PLASMA_enum nuplo = (uplo == PlasmaLower) ? PlasmaUpper : PlasmaLower ;

        dplasma_zlacpy( parsec, uplo,  Ainv,   (parsec_tiled_matrix_dc_t *)&A0 );
        dplasma_zlaset( parsec, nuplo, 0., 1., (parsec_tiled_matrix_dc_t *)&A0 );
    }
    dplasma_ztrmm(parsec, PlasmaLeft, uplo, PlasmaNoTrans, diag,
                  1., A, (parsec_tiled_matrix_dc_t *)&A0 );

    dplasma_zgeadd( parsec, PlasmaNoTrans,
                    -1.0, (parsec_tiled_matrix_dc_t*)&A0,
                     1.0, (parsec_tiled_matrix_dc_t*)&Id );

    Anorm    = dplasma_zlantr( parsec, PlasmaOneNorm, uplo, diag, A );
    Ainvnorm = dplasma_zlantr( parsec, PlasmaOneNorm, uplo, diag, Ainv );
    Rnorm    = dplasma_zlantr( parsec, PlasmaOneNorm, uplo, PlasmaNonUnit, (parsec_tiled_matrix_dc_t*)&Id );

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

    parsec_data_free(Id.mat);
    parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&Id);
    parsec_data_free(A0.mat);
    parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&A0);

    return info_solution;
}
