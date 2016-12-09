/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    double *work = NULL;
    double result;
    double normlap = 0.0;
    double normdag = 0.0;
    double eps = LAPACKE_dlamch_work('e');
    int iparam[IPARAM_SIZEOF];
    int An, i, u, ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 200, 200);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    An = dplasma_imax(M, N);
    LDA = max( LDA, M );

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, 1,
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDA, An, 0, 0,
                               M, An, SMB, SNB, 1));

    if( rank == 0 ) {
        work = (double *)malloc( max(M,N) * sizeof(double));
    }

    /*
     * General cases LANGE
     */
    {
        PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
            two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDA, N, 0, 0,
                                   M, N, SMB, SNB, P));

        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplrnt( parsec, 0, (tiled_matrix_desc_t *)&ddescA0, 3872);
        dplasma_zplrnt( parsec, 0, (tiled_matrix_desc_t *)&ddescA,  3872);
        if(loud > 2) printf("Done\n");

        for(i=0; i<4; i++) {
            if ( rank == 0 ) {
                printf("***************************************************\n");
            }
            if(loud > 2) printf("+++ Computing norm %s ... ", normsstr[i]);
            normdag = dplasma_zlange(parsec, norms[i],
                                     (tiled_matrix_desc_t *)&ddescA);

            if ( rank == 0 ) {
                normlap = LAPACKE_zlange_work(LAPACK_COL_MAJOR, normsstr[i][0], M, N,
                                              (parsec_complex64_t*)(ddescA0.mat), ddescA0.super.lm, work);
            }
            if(loud > 2) printf("Done.\n");

            if ( loud > 3 ) {
                printf( "%d: The norm %s of A is %e\n",
                        rank, normsstr[i], normdag);
            }

            if ( rank == 0 ) {
                result = fabs(normdag - normlap) / (normlap * eps) ;

                if ( loud > 3 ) {
                    printf( "%d: The norm %s of A is %e (LAPACK)\n",
                            rank, normsstr[i], normlap);
                }

                switch(norms[i]) {
                case PlasmaMaxNorm:
                    /* result should be perfectly equal */
                    break;
                case PlasmaInfNorm:
                    /* Sum order on the line can differ */
                    result = result / (double)N;
                    break;
                case PlasmaOneNorm:
                    /* Sum order on the column can differ */
                    result = result / (double)M;
                    break;
                case PlasmaFrobeniusNorm:
                    /* Sum order on every element can differ */
                    result = result / ((double)M * (double)N);
                    break;
                }

                if ( result < 1. ) {
                    printf(" ----- TESTING ZLANGE (%s) ... SUCCESS !\n", normsstr[i]);
                } else {
                    printf("       Ndag = %e, Nlap = %e\n", normdag, normlap );
                    printf("       | Ndag - Nlap | / Nlap = %e\n", result);
                    printf(" ----- TESTING ZLANGE (%s) ... FAILED !\n", normsstr[i]);
                    ret |= 1;
                }
            }
        }

        parsec_data_free(ddescA.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
    }

    /*
     * Triangular cases LANTR
     */
    {
        int d;

        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplrnt( parsec, 0., (tiled_matrix_desc_t *)&ddescA0, 3872);
        if(loud > 2) printf("Done\n");

        /* Computing the norm */
        PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
            two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDA, N, 0, 0,
                                   M, N, SMB, SNB, P));

        for(u=0; u<2; u++) {
            dplasma_zplrnt( parsec, 0., (tiled_matrix_desc_t *)&ddescA, 3872);

            for(d=0; d<2; d++) {
                for(i=0; i<4; i++) {
                    if ( rank == 0 ) {
                        printf("***************************************************\n");
                    }
                    if(loud > 2) printf("+++ Computing norm %s ... ", normsstr[i]);
                    normdag = dplasma_zlantr(parsec, norms[i], uplo[u], diag[d],
                                             (tiled_matrix_desc_t *)&ddescA);

                    if ( rank == 0 ) {
                        normlap = LAPACKE_zlantr_work(LAPACK_COL_MAJOR, normsstr[i][0], uplostr[u][0], diagstr[d][0], M, N,
                                                      (parsec_complex64_t*)(ddescA0.mat), ddescA0.super.lm, work);
                    }
                    if(loud > 2) printf("Done.\n");

                    if ( loud > 3 ) {
                        printf( "%d: The norm %s of A is %e\n",
                                rank, normsstr[i], normdag);
                    }

                    if ( rank == 0 ) {
                        result = fabs(normdag - normlap) / (normlap * eps);

                        if ( loud > 3 ) {
                            printf( "%d: The norm %s of A is %e (LAPACK)\n",
                                    rank, normsstr[i], normlap);
                        }

                        switch(norms[i]) {
                        case PlasmaMaxNorm:
                            /* result should be perfectly equal */
                            break;
                        case PlasmaInfNorm:
                            /* Sum order on the line can differ */
                            result = result / (double)N;
                            break;
                        case PlasmaOneNorm:
                            /* Sum order on the column can differ */
                            result = result / (double)M;
                            break;
                        case PlasmaFrobeniusNorm:
                            /* Sum oreder on every element can differ */
                            result = result / ((double)M * (double)N);
                            break;
                        }

                        if ( result < 1. ) {
                            printf(" ----- TESTING ZLANTR (%s, %s, %s) ... SUCCESS !\n",
                                   normsstr[i], uplostr[u], diagstr[d]);
                        } else {
                            printf(" ----- TESTING ZLANTR (%s, %s, %s) ... FAILED !\n",
                                   normsstr[i], uplostr[u], diagstr[d]);
                            printf("       | Ndag - Nlap | / Nlap = %e\n", normdag);
                            ret |= 1;
                        }
                    }
                }
            }
        }
        parsec_data_free(ddescA.mat);
        tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescA);
    }

    /* Let set N=M for the triangular cases */
    N = M;

    /*
     * Symmetric cases LANSY
     */
    {
        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplgsy( parsec, 0., PlasmaUpperLower, (tiled_matrix_desc_t *)&ddescA0, 3872);
        if(loud > 2) printf("Done\n");

        for(u=0; u<2; u++) {

            /* Computing the norm */
            PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
                sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                           nodes, rank, MB, NB, LDA, N, 0, 0,
                                           M, N, P, uplo[u]));

            dplasma_zplgsy( parsec, 0., uplo[u], (tiled_matrix_desc_t *)&ddescA, 3872);

            for(i=0; i<4; i++) {
                if ( rank == 0 ) {
                    printf("***************************************************\n");
                }
                if(loud > 2) printf("+++ Computing norm %s ... ", normsstr[i]);
                normdag = dplasma_zlansy(parsec, norms[i], uplo[u],
                                         (tiled_matrix_desc_t *)&ddescA);

                if ( rank == 0 ) {
                    normlap = LAPACKE_zlansy_work(LAPACK_COL_MAJOR, normsstr[i][0], uplostr[u][0], M,
                                                  (parsec_complex64_t*)(ddescA0.mat), ddescA0.super.lm, work);
                }
                if(loud > 2) printf("Done.\n");

                if ( loud > 3 ) {
                    printf( "%d: The norm %s of A is %e\n",
                            rank, normsstr[i], normdag);
                }

                if ( rank == 0 ) {
                    result = fabs(normdag - normlap) / (normlap * eps);

                    if ( loud > 3 ) {
                        printf( "%d: The norm %s of A is %e (LAPACK)\n",
                                rank, normsstr[i], normlap);
                    }

                    switch(norms[i]) {
                    case PlasmaMaxNorm:
                        /* result should be perfectly equal */
                        break;
                    case PlasmaInfNorm:
                        /* Sum order on the line can differ */
                        result = result / (double)N;
                        break;
                    case PlasmaOneNorm:
                        /* Sum order on the column can differ */
                        result = result / (double)M;
                        break;
                    case PlasmaFrobeniusNorm:
                        /* Sum oreder on every element can differ */
                        result = result / ((double)M * (double)N);
                        break;
                    }

                    if ( result < 1. ) {
                        printf(" ----- TESTING ZLANSY (%s, %s) ... SUCCESS !\n", uplostr[u], normsstr[i]);
                    } else {
                        printf(" ----- TESTING ZLANSY (%s, %s) ... FAILED !\n", uplostr[u], normsstr[i]);
                        printf("       | Ndag - Nlap | / Nlap = %e\n", normdag);
                        ret |= 1;
                    }
                }
            }

            parsec_data_free(ddescA.mat);
            tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
        }
    }

#if defined(PRECISION_z) || defined(PRECISION_c)
    /*
     * Hermitian cases LANHE
     */
    {
        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplghe( parsec, 0., PlasmaUpperLower, (tiled_matrix_desc_t *)&ddescA0, 3872);
        if(loud > 2) printf("Done\n");

        for(u=0; u<2; u++) {

            /* Computing the norm */
            PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
                sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                           nodes, rank, MB, NB, LDA, N, 0, 0,
                                           M, N, P, uplo[u]));

            dplasma_zplghe( parsec, 0., uplo[u], (tiled_matrix_desc_t *)&ddescA, 3872);

            for(i=0; i<4; i++) {
                if ( rank == 0 ) {
                    printf("***************************************************\n");
                }
                if(loud > 2) printf("+++ Computing norm %s ... ", normsstr[i]);
                normdag = dplasma_zlanhe(parsec, norms[i], uplo[u],
                                         (tiled_matrix_desc_t *)&ddescA);

                if ( rank == 0 ) {
                    normlap = LAPACKE_zlanhe_work(LAPACK_COL_MAJOR, normsstr[i][0], uplostr[u][0], M,
                                                  (parsec_complex64_t*)(ddescA0.mat), ddescA0.super.lm, work);
                }
                if(loud > 2) printf("Done.\n");

                if ( loud > 3 ) {
                    printf( "%d: The norm %s of A is %e\n",
                            rank, normsstr[i], normdag);
                }

                if ( rank == 0 ) {
                    result = fabs(normdag - normlap) / (normlap * eps);

                    if ( loud > 3 ) {
                        printf( "%d: The norm %s of A is %e (LAPACK)\n",
                                rank, normsstr[i], normlap);
                    }
                    switch(norms[i]) {
                    case PlasmaMaxNorm:
                        /* result should be perfectly equal */
                        break;
                    case PlasmaInfNorm:
                        /* Sum order on the line can differ */
                        result = result / (double)N;
                        break;
                    case PlasmaOneNorm:
                        /* Sum order on the column can differ */
                        result = result / (double)M;
                        break;
                    case PlasmaFrobeniusNorm:
                        /* Sum oreder on every element can differ */
                        result = result / ((double)M * (double)N);
                        break;
                    }

                    if ( result < 1. ) {
                        printf(" ----- TESTING ZLANHE (%s, %s) ... SUCCESS !\n", uplostr[u], normsstr[i]);
                    } else {
                        printf(" ----- TESTING ZLANHE (%s, %s) ... FAILED !\n", uplostr[u], normsstr[i]);
                        printf("       | Ndag - Nlap | / Nlap = %e\n", normdag);
                        ret |= 1;
                    }
                }
            }

            parsec_data_free(ddescA.mat);
            tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
        }
    }
#endif

    if ( rank == 0 ) {
        printf("***************************************************\n");
        free( work );
    }
    parsec_data_free(ddescA0.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0);

    cleanup_parsec(parsec, iparam);

    return ret;
}
