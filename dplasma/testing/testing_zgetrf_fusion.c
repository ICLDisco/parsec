/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

//#define MYDEBUG 1
static int check_solution( parsec_context_t *parsec, int loud,
                           parsec_tiled_matrix_dc_t *dcA,
                           parsec_tiled_matrix_dc_t *dcB,
                           parsec_tiled_matrix_dc_t *dcX );

int dplasma_iprint( parsec_context_t *parsec,
                    PLASMA_enum uplo,
                    parsec_tiled_matrix_dc_t *A);

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int info = 0;
    int ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 200, 200);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
#if defined(HAVE_CUDA)
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize Parsec */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZGETRF, ((DagDouble_t)M, (DagDouble_t)N));

#ifndef MYDEBUG
    if ( M != N && check ) {
        fprintf(stderr, "Check cannot be perfomed with M != N\n");
        check = 0;
    }
#endif
    LDA = dplasma_imax( LDA, MT * MB );

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(dcIPIV, 1,
        two_dim_block_cyclic, (&dcIPIV, matrix_Integer, matrix_Tile,
                               nodes, rank, 1, NB, P, dplasma_imin(M, N), 0, 0,
                               P, dplasma_imin(M, N), SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(dcA0, check,
        two_dim_block_cyclic, (&dcA0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(dcB, check,
        two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(dcX, check,
        two_dim_block_cyclic, (&dcX, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

#ifdef MYDEBUG
    PASTE_CODE_ALLOCATE_MATRIX(dcAl, check,
                               two_dim_block_cyclic, (&dcAl, matrix_ComplexDouble, matrix_Lapack,
                                                      1, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, 1));

    PASTE_CODE_ALLOCATE_MATRIX(dcIPIVl, check,
                               two_dim_block_cyclic, (&dcIPIVl, matrix_Integer, matrix_Lapack,
                                                      1, rank, 1, NB, 1, dplasma_imin(M, N), 0, 0,
                                                      1, dplasma_imin(M, N), SMB, SNB, 1));
#endif

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, 7657);

    /* Increase diagonale to avoid pivoting */
    if (0)
    {
        parsec_tiled_matrix_dc_t *dcA = (parsec_tiled_matrix_dc_t *)&dcA;
        int minmnt = dplasma_imin( dcA->mt, dcA->nt );
        int minmn  = dplasma_imin( dcA->m,  dcA->n );
        int t, e;

        for(t = 0; t < minmnt; t++ ) {
          if(((parsec_dc_t*) &dcA)->rank_of(((parsec_dc_t*) &dcA), t, t)  == ((parsec_dc_t*) &dcA)->myrank)
            {
              parsec_data_t* data = ((parsec_dc_t*) &dcA)->data_of(((parsec_dc_t*) &dcA), t, t);
              parsec_data_copy_t* copy = parsec_data_get_copy(data, 0);
              parsec_complex64_t *tab = (parsec_complex64_t*)parsec_data_copy_get_ptr(copy);
              for(e = 0; e < dcA->mb; e++)
                tab[e * dcA->mb + e] += (parsec_complex64_t)minmn;
            }
        }
    }

    if ( check )
    {
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcA,
                        (parsec_tiled_matrix_dc_t *)&dcA0 );
#ifdef MYDEBUG
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcA,
                        (parsec_tiled_matrix_dc_t *)&dcAl );
#endif
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcB, 2354);
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcB,
                        (parsec_tiled_matrix_dc_t *)&dcX );
    }
    if(loud > 2) printf("Done\n");

    /* Create Parsec */
    if(loud > 2) printf("+++ Computing getrf ... ");
    PASTE_CODE_ENQUEUE_KERNEL(parsec, zgetrf_fusion,
                              ((parsec_tiled_matrix_dc_t*)&dcA,
                               (parsec_tiled_matrix_dc_t*)&dcIPIV,
                               P,
                               Q,
                               &info));
    /* lets rock! */
    PASTE_CODE_PROGRESS_KERNEL(parsec, zgetrf_fusion);
    dplasma_zgetrf_fusion_Destruct( PARSEC_zgetrf_fusion );
    if(loud > 2) printf("Done.\n");

    if ( info != 0 ) {
        if( rank == 0 && loud ) printf("-- Factorization is suspicious (info = %d) ! \n", info );
        ret |= 1;
    }
    else if ( check ) {
#ifdef MYDEBUG
        if( rank  == 0 ) {
            int i;
            LAPACKE_zgetrf_work(LAPACK_COL_MAJOR, M, N,
                                (parsec_complex64_t*)(dcAl.mat), LDA,
                                (int *)(dcIPIVl.mat));

            printf("The Lapack swap are :\n");
            for(i=0; i < dplasma_imin(M, N); i++) {
                if ( i%NB == 0 )
                    printf("\n(%d, %d) ", 0, i/NB );
                printf( "%d ", ((int *)dcIPIVl.mat)[i] );
            }
            printf("\n");
        }
/*         dplasma_iprint(parsec, PlasmaUpperLower, (parsec_tiled_matrix_dc_t*)&dcIPIV);  */
        dplasma_zprint(parsec, PlasmaUpperLower, (parsec_tiled_matrix_dc_t*)&dcA);
        dplasma_zprint(parsec, PlasmaUpperLower, (parsec_tiled_matrix_dc_t*)&dcAl);

        dplasma_zgeadd( parsec, PlasmaUpperLower, -1.0,
                        (parsec_tiled_matrix_dc_t*)&dcA,
                        (parsec_tiled_matrix_dc_t*)&dcAl );
        dplasma_zprint(parsec, PlasmaUpperLower, (parsec_tiled_matrix_dc_t*)&dcAl);

        parsec_data_free(dcAl.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcAl);
        parsec_data_free(dcIPIVl.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcIPIVl);
#else
        dplasma_ztrsmpl_fusion(parsec,
                               (parsec_tiled_matrix_dc_t *)&dcA,
                               (parsec_tiled_matrix_dc_t *)&dcIPIV,
                               (parsec_tiled_matrix_dc_t *)&dcX);

        dplasma_ztrsm(parsec, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit,
                      1.0, (parsec_tiled_matrix_dc_t *)&dcA,
                           (parsec_tiled_matrix_dc_t *)&dcX);

        /* Check the solution */
        ret |= check_solution( parsec, (rank == 0) ? loud : 0,
                               (parsec_tiled_matrix_dc_t *)&dcA0,
                               (parsec_tiled_matrix_dc_t *)&dcB,
                               (parsec_tiled_matrix_dc_t *)&dcX);
#endif
    }

    if ( check ) {
        parsec_data_free(dcA0.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA0);
        parsec_data_free(dcB.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB);
        parsec_data_free(dcX.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcX);
    }

#if defined(HAVE_CUDA)
    if(iparam[IPARAM_NGPUS] > 0) {
        parsec_gpu_data_unregister();
        parsec_gpu_kernel_fini(parsec, "zgemm");
    }
#endif
    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcA);
    parsec_data_free(dcIPIV.mat);
    parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcIPIV);

    cleanup_parsec(parsec, iparam);

    return ret;
}



static int check_solution( parsec_context_t *parsec, int loud,
                           parsec_tiled_matrix_dc_t *dcA,
                           parsec_tiled_matrix_dc_t *dcB,
                           parsec_tiled_matrix_dc_t *dcX )
{
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    int m = dcB->m;
    double eps = LAPACKE_dlamch_work('e');

    Anorm = dplasma_zlange(parsec, PlasmaInfNorm, dcA);
    Bnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcB);
    Xnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcX);

    /* Compute b - A*x */
    dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans, -1.0, dcA, dcX, 1.0, dcB);

    Rnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcB);

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * m * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||X||_oo = %e, ||B||_oo= %e, ||A X - B||_oo = %e\n",
                    Anorm, Xnorm, Bnorm, Rnorm );

        printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", result);
    }

    if (  isnan(Xnorm) || isinf(Xnorm) || isnan(result) || isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
        if( loud ) printf("-- Solution is CORRECT ! \n");
        info_solution = 0;
    }

    return info_solution;
}
